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
    if tl.program_id(0) != 0 or tl.program_id(1) != 0:
        return
    # b = 2, h = 96,
    # tl.static_print("b=", b, " h=", h, " n=", n, " d=", d, " e=", e, " BLOCK=", BLOCK, " NUM_BLOCK=", NUM_BLOCK, " BLOCK_MODEL=", BLOCK_MODEL)
    bx = tl.program_id(0)  # bh offset
    by = tl.program_id(1)  # e offset

    print(f"Q: {Q}, K: {K}, V: {V}")

    q_head_val = tl.load(Q + tl.arange(0, 256)).to(tl.float32)
    print(f"q_head_val={q_head_val}")

    # tl.device_print("bx=", bx, " by=", by)
    # print(f"bx={bx}, by={by}")

    bh_offset = bx * n * d
    h_id = bx % h

    Q += bh_offset
    K += bh_offset
    V = V + bx * n * e
    O = O + bx * n * e

    # tl.device_print("fwd_kernel_v1 q: ", Q)

    kv = tl.zeros((d, BLOCK_MODEL), dtype=tl.float32) # [d, BLOCK_MODEL]

    # calculate decay
    slope = tl.load(S + h_id).to(tl.float32)
    q_decay = tl.exp((-slope * tl.arange(0, BLOCK))[:, None])  # BLOCK x 1
    k_pe = BLOCK - tl.arange(0, BLOCK)
    k_decay = tl.exp(-slope * k_pe[None, :])  # 1 x BLOCK
    block_decay = tl.exp(-slope * BLOCK)
    index = tl.arange(0, BLOCK)[:, None] - tl.arange(0, BLOCK)[None, :]
    s_index = slope * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    # tl.static_print("s_index shape=", s_index.shape)
    # print(f"index={index}, slope={slope}")
    # print(f"s_index={s_index}")

    diag_decay = tl.exp(s_index)  # BLOCK x BLOCK
    # print(f"diag_decay={diag_decay}")

    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        print(f"s_index: {s_index}, diag_decay: {diag_decay}")

    for i in range(NUM_BLOCK):

        # load q, size: BLOCK x d
        q_row_off = tl.arange(0, BLOCK) + i * BLOCK
        q_col_off = tl.arange(0, d)
        q_row_mask = q_row_off < n
        q_off = q_row_off[:, None] * d + q_col_off[None, :]
        if i == 0:
            print(f"q_off={Q + q_off}")
        q = tl.load(Q + q_off, mask=q_row_mask[:, None], other=0.0).to(tl.float32)

        if tl.program_id(0) == 0 and tl.program_id(1) == 0 and i == 0:
            print(f"q: {q}")

        # tl.static_print(f"q shape=", q_off.shape)

        # tl.device_print("fwd_kernel_v1 q: ", q)

        # load k^T size: d x BLOCK
        kt_row_off = tl.arange(0, d)
        kt_col_off = tl.arange(0, BLOCK) + i * BLOCK
        kt_col_off_mask = kt_col_off < n
        kt_off = kt_col_off[None, :] * d + kt_row_off[:, None]
        kt = tl.load(K + kt_off, mask=kt_col_off_mask[None, :], other=0.0).to(tl.float32)

        # tl.device_print("fwd_kernel_v1 kt: ", kt)

        # tl.static_print(f"kt shape=", kt_off.shape)

        # load V size BLOCK x BLOCK_MODEL
        v_row_off = tl.arange(0, BLOCK) + i * BLOCK
        v_col_off = tl.arange(0, BLOCK_MODEL) + by * BLOCK_MODEL
        v_row_mask = v_row_off < n
        v_off = v_row_off[:, None] * e + v_col_off[None, :]
        v = tl.load(V + v_off, mask=v_row_mask[:, None], other=0.0).to(tl.float32)


        if tl.program_id(0) == 0 and tl.program_id(1) == 0 and i == 0:
            print(f"q: {q}, k: {kt}, v{v}")

        # tl.device_print("fwd_kernel_v1 v: ", v)

        # tl.static_print(f"v shape=", v.shape)

        # compute intra block
        qk = tl.dot(q, kt)  # BLOCK x BLOCK
        o_intra = tl.dot(qk * diag_decay, v)  # o_intra = qkv, size: BLOCK x BLOCK_MODEL
        #tl.static_print("fwd_kernel_v1: o_intra shape=", o_intra.shape)
        # tl.device_print("fwd_kernel_v1 o_intra: ", o_intra)
        # compute inter block

        o_inter = tl.dot(q * q_decay, kv) # BLOCK x BLOCK_MODEL
        # tl.device_print("fwd_kernel_v1 o_inter: ", o_inter)
        # tl.static_print("fwd_kernel_v1: o_inter shape=", o_inter.shape)

        if tl.program_id(0) == 0 and tl.program_id(1) == 0 and i == 0:
            print(f"o_intra {o_intra}, o_inter {o_inter}")

        o = o_intra + o_inter
        #tl.device_print("fwd_kernel_v1 o value: ", o)


        # tl.static_print("fwd_kernel_v1: o shape=", o.shape)

        # update kv
        new_kv = tl.dot(kt * k_decay, v)  # d x BLOCK_MODEL

        kv = block_decay * kv + new_kv

        # write result back TODO: o data type align
        o_row_off = tl.arange(0, BLOCK) + i * BLOCK
        o_col_off = tl.arange(0, BLOCK_MODEL) + by * BLOCK_MODEL
        # tl.static_print(f"o_col_off shape=", o_col_off.shape)
        o_off = o_row_off[:, None] * e + o_col_off[None, :]
        # tl.static_print("fwd_kernel_v1: o_off shape=", o_off.shape)
        # tl.device_print("fwd_kernel_v1 o value: ", o)
        o_row_mask = o_row_off < n
        o_col_mask = o_col_off < e
        tl.store(O + o_off, o.to(O.dtype.element_ty), mask=o_row_mask[:, None] & o_col_mask[None, :])

    # tl.device_print("bx=", bx, " by=", by, " finished!")