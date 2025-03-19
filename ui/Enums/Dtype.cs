using System.ComponentModel;

namespace DiffusionPipeInterface.Enums
{
    public enum Dtype
    {
        [Description("bfloat16")]
        BFloat16,
        [Description("bfloat32")]
        BFloat32,
        [Description("float8")]
        Float8,
        [Description("float16")]
        Float16,
        [Description("float32")]
        Float32,
    }
}
