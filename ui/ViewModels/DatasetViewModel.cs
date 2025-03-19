using Tomlyn.Model;

namespace DiffusionPipeInterface.ViewModels
{
    public class DatasetViewModel
    {
        public int Key { get; set; }
        public string Name { get; set; } = null!;
        public string Path { get; set; } = null!;
        public TomlTable DatasetToml { get; set; } = null!;
        public TomlTable? ConfigurationToml { get; set; }
        public string? ConfigurationTomlString { get; set; }
    }
}
