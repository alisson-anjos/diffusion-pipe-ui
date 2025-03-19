using System.ComponentModel;

namespace DiffusionPipeInterface.Enums
{
    public enum SampleMethod
    {
        [Description("logit_normal")]
        LogitNormal,
        
        [Description("uniform")]
        Uniform
    }
}
