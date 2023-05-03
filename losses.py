import jax.lax as lax
import jax.numpy as jnp


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = lax.stop_gradient(rl)
            rl = rl.astype(jnp.float32)
            gl = gl.astype(jnp.float32)
            loss += jnp.mean(jnp.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.astype(jnp.float32)
        dg = dg.astype(jnp.float32)

        r_loss = jnp.mean((1 - dr) ** 2)
        g_loss = jnp.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.astype(jnp.float32)
        l = jnp.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p = z_p.astype(jnp.float32)
    logs_q = logs_q.astype(jnp.float32)
    m_p = m_p.astype(jnp.float32)
    logs_p = logs_p.astype(jnp.float32)
    z_mask = z_mask.astype(jnp.float32)

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * jnp.exp(-2.0 * logs_p)
    kl = jnp.sum(kl * z_mask)
    l = kl / jnp.sum(z_mask)
    return l
