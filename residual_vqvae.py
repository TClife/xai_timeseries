from math import sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast
from einops import rearrange, repeat, pack, unpack
from vector_quantize_pytorch import ResidualVQ, VectorQuantize

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, 1, padding = 0),
            nn.ReLU(),
            nn.Conv1d(chan, chan, 1, padding = 0),
            nn.ReLU(),
            nn.Conv1d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class VQVAE(nn.Module):
    def __init__(
        self,
        image_size = 1632,
        num_tokens = 1024,
        codebook_dim = 512,
        num_layers = 2,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 1,
        temperature = 0.9,
        straight_through = False,
        vq_decay = 0.8,
        commitment_weight = 1.
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.vq = VectorQuantize(
            dim = codebook_dim,
            codebook_size = num_tokens,
            decay = vq_decay,
            commitment_weight = commitment_weight
        )
        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []      
        
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv1d(enc_in, enc_out, 2, stride = 2, padding = 0), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose1d(dec_in, dec_out, 2, stride = 2, padding = 0), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv1d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv1d(enc_chans[-1], codebook_dim, 1))
        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        encoded = self.forward(images, return_encoded = True)
        encoded = rearrange(encoded, 'b c h w -> b (h w) c')
        _, indices, _ = self.vq(encoded)
        return indices

    def indices_to_codebook(self, indices):
        codebook = self.vq.get_codes_from_indices(indices)
        return codebook

    def decode(
        self,
        img_seq
    ):
        codebook = rearrange(self.vq.embed, 'd n -> n d')
        image_embeds = codebook[img_seq]
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = True,
        return_encoded = False
    ):
        shape, device = img.shape, img.device

        encoded = self.encoder(img.float()) #[bs, 3, 32, 32] --> [bs, 512, 4, 4]

        if return_encoded:
            return encoded

        t = encoded.shape[-1] #[4, 4]

        encoded = rearrange(encoded, 'b c t -> b t c') #[bs, 512, 4, 4] --> [bs, 16, 512]
        quantized, indices, commit_loss = self.vq(encoded)
        quantized = rearrange(quantized, 'b t c -> b c t', t = t) #[bs, 32, 12] == [batch, embedding_dim, timestep]
        out = self.decoder(quantized)

        if not return_loss:
            return out

        # reconstruction loss and VQ commitment loss

        recon_loss = F.mse_loss(img, out)
        

        return out, recon_loss, commit_loss, indices, quantized

class Residual_VQVAE(nn.Module):
    def __init__(
        self,
        image_size = 1632,
        num_tokens = 1024,
        codebook_dim = 512,
        num_layers = 2,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 1,
        temperature = 0.9,
        straight_through = False,
        vq_decay = 0.8,
        commitment_weight = 1.,
        num_quantizers = 2
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.vq = ResidualVQ(
            dim = codebook_dim,
            codebook_size = num_tokens,
            num_quantizers = num_quantizers,
        )
        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []      
        
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv1d(enc_in, enc_out, 2, stride = 2, padding = 0), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose1d(dec_in, dec_out, 2, stride = 2, padding = 0), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv1d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv1d(enc_chans[-1], codebook_dim, 1))
        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        encoded = self.forward(images, return_encoded = True)
        encoded = rearrange(encoded, 'b c t -> b t c')
        _, indices, _ = self.vq(encoded)
        return indices
    
    @torch.no_grad()
    @eval_decorator
    def indices_to_recon(self, indices):
        self.vq.eval()
        
        quantized_out = self.vq.get_codes_from_indices(indices)
        quantized_out = rearrange(quantized_out.sum(dim=0), 'b t c -> b c t', t = 13) #[bs, 32, 12] == [batch, embedding_dim, timestep]
        out = self.decoder(quantized_out)
        
        return out

    def decode(
        self,
        img_seq
    ):
        codebook = rearrange(self.vq.embed, 'd n -> n d')
        image_embeds = codebook[img_seq]
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b t d -> b d t')
        images = self.decoder(img_seq)
        return images

    def forward(
        self,
        img,
        return_loss = True,
        return_encoded = False
    ):
        shape, device = img.shape, img.device

        encoded = self.encoder(img.float()) #[bs, 3, 32, 32] --> [bs, 512, 4, 4]

        if return_encoded:
            return encoded

        t = encoded.shape[-1] #[4, 4]

        encoded = rearrange(encoded, 'b c t -> b t c') #[bs, 512, 4, 4] --> [bs, 16, 512]
        quantized, indices, commit_loss = self.vq(encoded)
        quantized = rearrange(quantized, 'b t c -> b c t', t = t) #[bs, 32, 12] == [batch, embedding_dim, timestep]
        out = self.decoder(quantized)

        if not return_loss:
            return out

        # reconstruction loss and VQ commitment loss

        recon_loss = F.mse_loss(img, out)
        

        return out, recon_loss, commit_loss, indices, quantized

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p = 2)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)

# distance types

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        use_ddp = False,
        learnable_codebook = False,
        sample_codebook_temp = 0
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4 #[bs, 16, 512]

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...') #[1, bs, 16, 512]

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d') #[1, bs * 16, 512] = [1, 2048, 512]

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach() #[1, 256, 512]

        dist = -torch.cdist(flatten, embed, p = 2) #[1, bs*16, 256] = [1, 2048, 256]

        embed_ind = gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp) #[1, 2048]
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype) #[1, 2048, 256]
        embed_ind = embed_ind.view(*shape[:-1]) #[1, 2048, 16]

        quantize = batched_embedding(embed_ind, self.embed) #[1, 128, 16], [1, 256, 512] ==> [1, 128, 16, 512]

        if self.training:
            cluster_size = embed_onehot.sum(dim = 1) #[1, 256]

            self.all_reduce_fn(cluster_size)
            self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot) #[1, 2048, 512] [1, 2048, 256] ==> [1, 256, 512] 
            self.all_reduce_fn(embed_sum.contiguous()) 
            self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind #[bs, 12, 32], [bs, 16]

class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        use_ddp = False,
        learnable_codebook = False,
        sample_codebook_temp = 0.
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)

        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            self.cluster_size.data.lerp_(bins, 1 - self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                embed,
                embed_normalized
            )

            self.embed.data.lerp_(embed_normalized, 1 - self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind

