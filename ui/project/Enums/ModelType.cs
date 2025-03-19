using System.ComponentModel;

namespace DiffusionPipeInterface.Enums
{
    public enum ModelType
    {
        [Description("sdxl")]
        SDXL,
        [Description("flux")]
        Flux,
        [Description("ltx-video")]
        LTX,
        [Description("hunyuan-video")]
        Hunyuan,
        [Description("cosmos")]
        Cosmos,
        [Description("lumina_2")]
        Lumina,
        [Description("wan")]
        Wan21,
        [Description("chroma")]
        Chroma,
    }
}
