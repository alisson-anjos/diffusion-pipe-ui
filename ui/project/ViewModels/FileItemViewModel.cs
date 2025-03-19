namespace DiffusionPipeInterface.ViewModels
{
    public class FileItemViewModel
    {
        public string Name { get; set; } = string.Empty;
        public bool IsDirectory { get; set; }
        public bool IsExpanded { get; set; }
        public bool IsSelected { get; set; }
        public string Icon { get; set; } = string.Empty;
        public string IconExpanded { get; set; } = string.Empty;
        public List<FileItemViewModel> Children { get; set; } = new List<FileItemViewModel>();
        public string FullPath { get; set; } = string.Empty;
    }
}
