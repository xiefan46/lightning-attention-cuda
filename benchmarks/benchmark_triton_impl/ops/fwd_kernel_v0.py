import triton
import triton.language as tl


# Adapted from https://github.com/OpenNLPLab/lightning-attention/blob/main/lightning_attn/ops/triton/lightning_attn2.py
@triton.jit
def fwd_kernel_v0(
        Q,
        K,
        V,
        Out,
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
    if tl.program_id(0) != 0 or tl.program_id(1) != 3:
        return

    print(f"b: {b}, h: {h}, n: {n}, d: {d}, e: {e}, BLOCK: {BLOCK}, NUM_BLOCK: {NUM_BLOCK}, BLOCK_MODEL: {BLOCK_MODEL}")

    # print(f"Q: {Q}, K: {K}, V: {V}")

    q_head_val = tl.load(Q + tl.arange(0, 256)).to(tl.float32)
    # print(f"q_head_val={q_head_val}")

    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]  # 1 x d



    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]  # d x 1
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]  # 1 x BLOCK_MODEL
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]  # 1 x BLOCK_MODEL
    S_block_ptr = S + off_h  # 1 x 1

    ##### init diag decay(Lambda); q, k decay; kv
    s = tl.load(S_block_ptr)  # 1 x 1
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    q_decay = tl.exp(-s.to(tl.float32) * off_block[:, None])  # s: 1x1 * off_block BLOCK x 1 = BLOCK x 1
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - off_block[None, :]))  # 1 x BLOCK
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)  # 1 x 1
    # diag decay
    index = off_block[:, None] - off_block[None, :]  # 相对位置 BLOCK x BLOCK
    s_index = s * index  # BLOCK * BLOCK
    s_index = tl.where(index >= 0, -s_index, float("-inf"))  # 下三角矩阵
    diag_decay = tl.exp(s_index)

    # if tl.program_id(0) == 0 and tl.program_id(1) == 0:
    #     print(f"s_index: {s_index}, diag_decay: {diag_decay}")

    # print(f"index={index}, slope={s}")
    # print(f"s_index={s_index}")
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)  # d x BLOCK_MODEL
    # tl.static_print("fwd_kernel_v0")
    # tl.static_print("NUM_BLOCK:", NUM_BLOCK)


    ##### compute
    for i in range(NUM_BLOCK):
        # load

        if i == 0:
            q_off = qk_offset + tl.arange(0, d)[None, :] + off_block[:, None] * d
            print(f"Q offset: {q_off}, Q mask: {off_block[:, None] < n}")

        q = tl.load(  # BLOCK * d
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # if i == 0:
        #     print(f"q_off={Q_block_ptr + off_block[:, None] * d}")
        #
        # if tl.program_id(0) == 0 and tl.program_id(1) == 0 and i == 0:
        #     print(f"q: {q}")

        k_trans = tl.load(  # k_trans -> d x BLOCK
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)

        if i == 0:
            print(f"K offset: {(qk_offset + tl.arange(0, d)[:, None]) + off_block[None, :] * d}, K mask: {off_block[None, :] < n}")

        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0  # BLOCK x BLOCK_MODEL
        ).to(tl.float32)

        if i == 0:
            print(f"V offset: {(v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]) + off_block[:, None] * e}, V mask: {off_block[:, None] < n}")

        # if tl.program_id(0) == 0 and tl.program_id(1) == 0 and i == 0:
        #     print(f"q: {q}, k: {k_trans}, v{v}")

        # compute
        qk = tl.dot(q, k_trans) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv) * q_decay

        # if tl.program_id(0) == 0 and tl.program_id(1) == 0 and i == 0:
        #     print(f"o_intra {o_intra}, o_inter {o_inter}")

        o = o_intra + o_inter

        o_off = (o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]) + off_block[:, None] * e

        if i == 0:
            print(f"O offset: {o_off}, O mask: {off_block[:, None] < n}")

        # tl.static_print("fwd_kernel_v0: o_off shape=", o_off.shape)
        # tl.device_print("fwd_kernel_v0 o value: ", o)
        # save and update
        tl.store(
            Out + o_off,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        off_block += BLOCK