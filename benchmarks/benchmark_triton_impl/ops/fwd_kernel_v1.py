import triton
import triton.language as tl

# TODO: 精度要怎么处理
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
    h_id = bx % h

    Q += bh_offset
    K += bh_offset
    V = V + bx * n * e + by * BLOCK_MODEL
    O = O + bx * n * e + by * BLOCK_MODEL

    kv = tl.zeros((d, BLOCK_MODEL), dtype=tl.float32)

    # calculate decay
    slope = tl.load(S + h_id).to(tl.float32)
    q_decay = tl.exp((-slope * tl.arange(0, BLOCK))[:, None])  # BLOCK x 1
    k_pe = BLOCK - tl.arange(0, BLOCK)
    k_decay = tl.exp(-slope * k_pe[None, :])  # 1 x BLOCK
    block_decay = tl.exp(-slope * BLOCK)
    index = tl.arange(0, BLOCK)[:, None] - tl.arange(0, BLOCK)[None, :]
    s_index = slope * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)  # BLOCK x BLOCK

    for i in range(NUM_BLOCK):
        # load q, size: BLOCK x d
        q_row_off = tl.arange(0, BLOCK) + i * BLOCK
        q_col_off = tl.arange(0, d)
        q_row_mask = q_row_off < n
        q_off = q_row_off[:, None] * d + q_col_off[None, :]
        q = tl.load(Q + q_off, mask=q_row_mask[:, None], other=0.0).to(tl.float32)

        tl.static_print(f"q shape=", q_off.shape)

        # load k^T size: d x BLOCK
        kt_row_off = tl.arange(0, d)
        kt_col_off = tl.arange(0, BLOCK) * d
        kt_col_off_mask = kt_col_off < n
        kt_off = kt_row_off[:, None] + kt_col_off[None, :]
        kt = tl.load(K + kt_off, mask=kt_col_off_mask[None, :], other=0.0).to(tl.float32)

        tl.static_print(f"kt shape=", kt_off.shape)

        # load V size BLOCK x BLOCK_MODEL
        v_row_off = tl.arange(0, BLOCK) + i * BLOCK
        v_col_off = tl.arange(0, BLOCK_MODEL)
        v_row_mask = v_row_off < n
        v_off = v_row_off[:, None] * e + v_col_off[None, :]
        v = tl.load(V + v_off, mask=v_row_mask[:, None], other=0.0).to(tl.float32)

        # compute intra block
        qk = tl.dot(q, kt)  # BLOCK x BLOCK
        o_intra = tl.dot(qk * diag_decay, v)  # o_intra = qkv, size: BLOCK x BLOCK_MODEL

        # compute inter block

        o_inter = tl.dot(q * q_decay, kv) # BLOCK x BLOCK_MODEL
        o = o_intra + o_inter

        # update kv
        new_kv = tl.dot(kt * k_decay, v)  # d x BLOCK_MODEL

        kv = block_decay * kv + new_kv

        # write result back TODO: o data type align
        o_row_off = tl.arange(0, BLOCK) + i * BLOCK
        o_col_off = tl.arange(0, BLOCK_MODEL)
        o_off = o_row_off[:, None] * e + o_col_off[None, :]
        o_row_mask = o_row_off < n
        tl.store(O + o_off, o.to(O.dtype.element_ty), mask=o_row_mask[:, None])