# Regenerate ail_norm.cu after a joyomni_ops update: python deploy/xvideo/models/dit/gen_ail_norm.py
# Generate deploy/xvideo/models/dit/ail_norm.cu from joyomni_ops' fused_norm_scale_shift.cu: same math, plus a
# broadcast mode (one scale/shift row per `rows_per_mod` token rows) so callers don't materialize [M, N] copies.
from pathlib import Path
root = Path(__file__).resolve().parents[3]   # deploy/
s = (root / "joyomni_ops/csrc/fused_norm_scale_shift.cu").read_text()
def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old, s.count(old)); s = s.replace(old, new)
sub('#include "joyomni_ops.h"\n', '#include <torch/extension.h>\n#define JO_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")\n')
sub("namespace joyomni_ops {", "namespace ail_norm {")
sub("}  // namespace joyomni_ops", "}  // namespace ail_norm")
sub("    const T4* scale, const T4* shift, const int n, bool affine, float eps) {",
    "    const T4* scale, const T4* shift, const int n, const int rows_per_mod, bool affine, float eps) {")
sub("""  scale += offset;  // per-row
  shift += offset;""", """  const int mod_offset = (m_idx / rows_per_mod) * n_4;  // ai-livestream: one scale/shift row per rows_per_mod rows
  scale += mod_offset;
  shift += mod_offset;""")
sub("            const torch::Tensor& shift, torch::Tensor& y, int norm_type, bool affine, float eps) {",
    "            const torch::Tensor& shift, torch::Tensor& y, int norm_type, bool affine, float eps, int rows_per_mod) {")
sub("(const T4*)scale.data_ptr(), (const T4*)shift.data_ptr(), (int)N, affine, eps)",
    "(const T4*)scale.data_ptr(), (const T4*)shift.data_ptr(), (int)N, rows_per_mod, affine, eps)")
sub("    const torch::Tensor& shift, int64_t norm_type, double eps) {",
    "    const torch::Tensor& shift, int64_t norm_type, double eps, int64_t rows_per_mod) {")
sub('  TORCH_CHECK(scale.size(0) == M && shift.size(0) == M, "scale/shift rows must equal M (per-row modulation)");',
    '  TORCH_CHECK(rows_per_mod >= 1 && scale.size(0) * rows_per_mod == M && shift.size(0) * rows_per_mod == M,\n'
    '              "scale/shift rows * rows_per_mod must equal M");\n'
    '  TORCH_CHECK(scale.is_contiguous() && shift.is_contiguous(), "scale/shift must be contiguous");')
sub("shift, y, (int)norm_type, affine, (float)eps);", "shift, y, (int)norm_type, affine, (float)eps, (int)rows_per_mod);", 3)
s += '''
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_norm_scale_shift", &ail_norm::fused_norm_scale_shift,
        "Norm(x) * (1 + scale) + shift, scale/shift [M / rows_per_mod, N] (joyomni_ops math, broadcast rows)");
}
'''
out = root / "xvideo/models/dit/ail_norm.cu"
out.write_text("// GENERATED from joyomni_ops/csrc/fused_norm_scale_shift.cu by the ai-livestream fork (see ail_ops.py).\n"
               "// Only change: scale/shift row index = row / rows_per_mod (broadcast adaLN modulation). Math untouched.\n" + s)
print("wrote", out)
