import sys
from pathlib import Path
import numpy as np

# Add gguf-py to path
sys.path.insert(0, str(Path("e:/llama.cpp study/llama.cpp/gguf-py")))
from gguf import GGUFWriter, GGMLQuantizationType

def get_tensor_nbytes(ne, dtype):
    if dtype == GGMLQuantizationType.Q4_K:
        blck_size = 256
        type_size = 144
    elif dtype == GGMLQuantizationType.F32:
        blck_size = 1
        type_size = 4
    elif dtype == GGMLQuantizationType.F16:
        blck_size = 1
        type_size = 2
    elif dtype == GGMLQuantizationType.Q8_0:
        blck_size = 32
        type_size = 34
    else:
        blck_size = 1
        type_size = 4
    
    n_elems = 1
    for d in ne:
        n_elems *= d
    return (n_elems // blck_size) * type_size

def create_qwen35moe_skeleton(output_path="qwen36_35b_skeleton.gguf"):
    writer = GGUFWriter(output_path, "qwen35moe")

    # Hyperparameters
    n_embd = 2048
    n_layer = 40
    n_head = 16
    n_head_kv = 2
    head_dim = 256
    n_ff = 512
    n_expert = 256
    n_expert_used = 8
    n_ff_exp = 512
    n_ff_shexp = 512
    full_attn_interval = 4
    ssm_d_conv = 4
    ssm_d_state = 128
    ssm_dt_rank = 32
    ssm_n_group = 16
    vocab_size = 248320

    writer.add_name("Qwen3.6-35B-A3B")
    writer.add_context_length(65536)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    
    # Custom architecture metadata
    writer.add_uint32("qwen35moe.expert_count", n_expert)
    writer.add_uint32("qwen35moe.expert_used_count", n_expert_used)
    writer.add_uint32("qwen35moe.expert_feed_forward_length", n_ff_exp)
    writer.add_uint32("qwen35moe.expert_shared_feed_forward_length", n_ff_shexp)
    writer.add_uint32("qwen35moe.full_attention_interval", full_attn_interval)
    writer.add_uint32("qwen35moe.ssm.inner_size", 4096)
    writer.add_uint32("qwen35moe.ssm.conv_kernel", ssm_d_conv)
    writer.add_uint32("qwen35moe.ssm.state_size", ssm_d_state)
    writer.add_uint32("qwen35moe.ssm.time_step_rank", ssm_dt_rank)
    writer.add_uint32("qwen35moe.ssm.group_count", ssm_n_group)
    writer.add_array("qwen35moe.rope.dimension_sections", [64, 64, 0, 0])

    # Vocabulary
    writer.add_tokenizer_model("gpt2")
    tokens = [f"<t{i}>" for i in range(vocab_size)]
    writer.add_token_list(tokens)
    writer.add_token_types([1] * vocab_size)
    writer.add_array("tokenizer.ggml.merges", ["<t0> <t1>"])

    dummy_f16 = np.zeros(1, dtype=np.float16)

    # In gguf-py, add_tensor(name, tensor, raw_shape=[dim_outer, dim_inner]) reverses to ne = [dim_inner, dim_outer].
    # So to get ne = [ne0, ne1], raw_shape must be [ne1, ne0].
    # To get ne = [ne0, ne1, ne2], raw_shape must be [ne2, ne1, ne0].

    tensor_specs = []

    def register_tensor(name, ne, dtype):
        tensor_specs.append((name, ne, dtype))
        raw_shape = list(reversed(ne))
        writer.add_tensor(name, dummy_f16, raw_shape=raw_shape, raw_dtype=dtype)

    # Global tensors
    register_tensor("token_embd.weight", [n_embd, vocab_size], GGMLQuantizationType.Q4_K)
    register_tensor("output_norm.weight", [n_embd], GGMLQuantizationType.F32)
    register_tensor("output.weight", [n_embd, vocab_size], GGMLQuantizationType.Q4_K)

    # Layer tensors
    head_k_dim = ssm_d_state
    head_v_dim = ssm_d_state
    n_k_heads = ssm_n_group
    n_v_heads = ssm_dt_rank
    key_dim = head_k_dim * n_k_heads
    value_dim = head_v_dim * n_v_heads
    conv_dim = key_dim * 2 + value_dim

    for il in range(n_layer):
        register_tensor(f"blk.{il}.attn_norm.weight", [n_embd], GGMLQuantizationType.F32)
        register_tensor(f"blk.{il}.post_attention_norm.weight", [n_embd], GGMLQuantizationType.F32)

        is_full_attn = ((il + 1) % full_attn_interval == 0)
        if is_full_attn:
            register_tensor(f"blk.{il}.attn_q.weight", [n_embd, head_dim * n_head * 2], GGMLQuantizationType.Q4_K)
            register_tensor(f"blk.{il}.attn_k.weight", [n_embd, head_dim * n_head_kv], GGMLQuantizationType.Q4_K)
            register_tensor(f"blk.{il}.attn_v.weight", [n_embd, head_dim * n_head_kv], GGMLQuantizationType.Q4_K)
            register_tensor(f"blk.{il}.attn_output.weight", [head_dim * n_head, n_embd], GGMLQuantizationType.Q4_K)
            register_tensor(f"blk.{il}.attn_q_norm.weight", [head_dim], GGMLQuantizationType.F32)
            register_tensor(f"blk.{il}.attn_k_norm.weight", [head_dim], GGMLQuantizationType.F32)
        else:
            register_tensor(f"blk.{il}.attn_qkv.weight", [n_embd, key_dim * 2 + value_dim], GGMLQuantizationType.Q4_K)
            register_tensor(f"blk.{il}.attn_gate.weight", [n_embd, value_dim], GGMLQuantizationType.Q4_K)
            register_tensor(f"blk.{il}.ssm_conv1d.weight", [ssm_d_conv, conv_dim], GGMLQuantizationType.F32)
            register_tensor(f"blk.{il}.ssm_dt.bias", [ssm_dt_rank], GGMLQuantizationType.F32)
            register_tensor(f"blk.{il}.ssm_a", [ssm_dt_rank], GGMLQuantizationType.F32)
            register_tensor(f"blk.{il}.ssm_beta.weight", [n_embd, n_v_heads], GGMLQuantizationType.F16)
            register_tensor(f"blk.{il}.ssm_alpha.weight", [n_embd, n_v_heads], GGMLQuantizationType.F16)
            register_tensor(f"blk.{il}.ssm_norm.weight", [head_v_dim], GGMLQuantizationType.F32)
            register_tensor(f"blk.{il}.ssm_out.weight", [value_dim, n_embd], GGMLQuantizationType.Q4_K)

        # MoE FFN
        register_tensor(f"blk.{il}.ffn_gate_inp.weight", [n_embd, n_expert], GGMLQuantizationType.F32)
        register_tensor(f"blk.{il}.ffn_down_exps.weight", [n_ff_exp, n_embd, n_expert], GGMLQuantizationType.Q4_K)
        register_tensor(f"blk.{il}.ffn_gate_exps.weight", [n_embd, n_ff_exp, n_expert], GGMLQuantizationType.Q4_K)
        register_tensor(f"blk.{il}.ffn_up_exps.weight", [n_embd, n_ff_exp, n_expert], GGMLQuantizationType.Q4_K)

        # Shared expert
        register_tensor(f"blk.{il}.ffn_gate_inp_shexp.weight", [n_embd], GGMLQuantizationType.F32)
        register_tensor(f"blk.{il}.ffn_gate_shexp.weight", [n_embd, n_ff_shexp], GGMLQuantizationType.Q4_K)
        register_tensor(f"blk.{il}.ffn_up_shexp.weight", [n_embd, n_ff_shexp], GGMLQuantizationType.Q4_K)
        register_tensor(f"blk.{il}.ffn_down_shexp.weight", [n_ff_shexp, n_embd], GGMLQuantizationType.Q4_K)

    # Now fix ti.nbytes for each tensor so that the written offset matches ggml expected offset
    print("Adjusting tensor info offsets...")
    for name, ne, dtype in tensor_specs:
        real_nbytes = get_tensor_nbytes(ne, dtype)
        writer.tensors[0][name].nbytes = real_nbytes

    print("Writing GGUF header and metadata...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Calculate total file size and seek to end to make file sparse/virtually sized
    data_start = writer.fout[0].tell()
    # Align data start to 32 bytes
    pad = GGUFWriter.ggml_pad(data_start, writer.data_alignment) - data_start
    if pad != 0:
        writer.fout[0].write(bytes([0] * pad))
        data_start += pad

    total_data_bytes = sum(GGUFWriter.ggml_pad(ti.nbytes, writer.data_alignment) for ti in writer.tensors[0].values())
    total_file_size = data_start + total_data_bytes

    print(f"Header size: {data_start / 1024 / 1024:.2f} MiB, Virtual model size: {total_file_size / 1024 / 1024 / 1024:.2f} GiB")
    
    # Seek to end of file to make Windows create a sparse virtual file (takes 0 actual disk space)
    writer.fout[0].seek(total_file_size - 1)
    writer.fout[0].write(b'\0')
    writer.close()
    print(f"Done! Skeleton GGUF created at {output_path}")

if __name__ == "__main__":
    create_qwen35moe_skeleton()
