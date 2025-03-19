namespace DiffusionPipeInterface.ViewModels
{
    public class DownloadRequest
    {
        public List<string> SelectedFiles { get; set; } = new List<string>();
        public string BaseDirectory { get; set; } = string.Empty;
    }
}
