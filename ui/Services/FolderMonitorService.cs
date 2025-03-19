namespace DiffusionPipeInterface.Services
{
    public class FolderMonitorService : IDisposable
    {
        private FileSystemWatcher _watcher;
        private string _folderPath;

        public event Action<string> OnNewSubfolderCreated;

        public void SetFolderPath(string folderPath)
        {
            if (_folderPath != folderPath)
            {
                _folderPath = folderPath;
                InitializeWatcher();
            }
        }

        private void InitializeWatcher()
        {
            // Dispose do watcher anterior, se existir
            _watcher?.Dispose();

            if (Directory.Exists(_folderPath))
            {
                _watcher = new FileSystemWatcher
                {
                    Path = _folderPath,
                    NotifyFilter = NotifyFilters.DirectoryName,
                    IncludeSubdirectories = false
                };

                _watcher.Created += OnCreated;
                _watcher.EnableRaisingEvents = true;
            }
        }

        private void OnCreated(object sender, FileSystemEventArgs e)
        {
            if (Directory.Exists(e.FullPath))
            {
                var safetensorsFiles = Directory.GetFiles(e.FullPath, "*.safetensors");
                if (safetensorsFiles.Length > 0)
                {
                    OnNewSubfolderCreated?.Invoke(safetensorsFiles[0]);
                }
            }
        }

        public void Dispose()
        {
            _watcher?.Dispose();
        }
    }
}
