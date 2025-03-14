import triton
import triton.language as tl


@triton.jit
def fwd_kernel_v1(
        Q,
        K,
        V,
        O,
        S,  # log lambda
        b: tl.constexpr,
        h: tl.constexpr,
        n: tl.constexpr,
        d: tl.constexpr,
        e: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_BLOCK: tl.constexpr,
        BLOCK_MODEL: tl.constexpr,
):
    bx = tl.program_id(0)  # bh offset
    by = tl.program_id(1)  # e offset

    bh_offset = bx * n * d
    h_id = bh_offset % h

    Q += bh_offset
    K += bh_offset
    V = V + bx * n * e + by * BLOCK_MODEL
    O = O + bx * n * e + by * BLOCK_MODEL

    kv = tl.zeros((d, BLOCK_MODEL))

    # calculate decay
    slope = tl.load(S + h_id)
    q_decay = tl.exp((tl.arange(BLOCK) * slope)[:, None])  # BLOCK x 1
    k_decay = tl.exp(((BLOCK - tl.arange(BLOCK)) * slope)[None, 1])  # 1 x BLOCK
    block_decay = tl.exp(slope * BLOCK)
    index = tl.arange(BLOCK)[:, None] - tl.arange(BLOCK)[None, :]
    s_index = slope * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)  # BLOCK x BLOCK

    for i in range(NUM_BLOCK):
        # load q, size: BLOCK x d
        q_row_off = tl.arange(BLOCK) + i * BLOCK
        q_col_off = tl.arange(d)
        q_row_mask = q_row_off < n
        q_col_mask = q_col_off < d
        q_off = q_row_off[:, None] & d + q_col_off[None, :]
        q = tl.load(Q + q_off, mask=q_row_mask[:, None] * q_col_mask[None, :], other=0.0)

        # load k^T size: d x BLOCK
        kt_row_off = tl.arange(d)
        kt_col_off = tl.arange(BLOCK) * d
        kt_row_off_mask = kt_row_off < d
        kt_col_off_mask = kt_col_off < n
        kt_off = kt_row_off[:, None] + kt_col_off[None, :]
        kt_mask = kt_row_off_mask[:, None] & kt_col_off_mask[None, :]
        kt = tl.load(K + kt_off, mask=kt_mask, other=0.0)

        # load V size BLOCK x BLOCK_MODEL
        v_row_off = tl.arange(BLOCK)
        v_col_off = tl.arange(BLOCK_MODEL)
        v_row_mask = v_row_off < n
        v_col_mask = v_col_off < e
        v_off = v_row_off[:, None] * e + v_col_off[None, :]
        v_mask = v_row_mask[:, None] & v_col_mask[None, :]
        v = tl.load(V + v_off, mask=v_mask, other=0.0)

        # compute intra block
        qk = q @ kt  # BLOCK x BLOCK
        o_intra = (qk * diag_decay) @ v  # o_intra = qkv, size: BLOCK x BLOCK_MODEL

        # compute inter block
        o_inter = (q * q_decay) @ kv  # BLOCK x BLOCK_MODEL

        o = o_intra + o_inter

        # update kv
        new_kv = (kt * k_decay) @ v   # d x BLOCK_MODEL
        kv = block_decay * kv + new_kv

        # write result back
        o_row_off = tl.arange(BLOCK)
        o_col_off = tl.arange(BLOCK_MODEL)
        o_off = o_row_off[:, None] * e + o_col_off[None, :]
        o_row_mask = o_row_off < n
        o_col_mask = o_col_off < e
        o_mask = o_row_mask[:, None] & o_col_mask[None, :]
        tl.store(O + o_off, o, mask=o_mask)