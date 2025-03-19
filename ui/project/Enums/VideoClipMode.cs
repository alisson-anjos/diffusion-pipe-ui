using System.ComponentModel;

namespace DiffusionPipeInterface.Enums
{
    public enum VideoClipMode
    {
        [Description("single_beginning")]
        SingleBeginning,
        [Description("single_middle")]
        SingleMiddle,
        [Description("multiple_overlapping")]
        MultipleOverllaping
    }
}
