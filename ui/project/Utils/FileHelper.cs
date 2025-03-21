
using DiffusionPipeInterface.Models;
using DiffusionPipeInterface.ViewModels;
using Microsoft.EntityFrameworkCore.Metadata.Internal;
using Tomlyn;

namespace DiffusionPipeInterface.Utils
{
    public static class FileHelper
    {

        public static HashSet<string> ImageExtensions { get; set; } = new HashSet<string> {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heif", ".heic", ".svg", ".ico", ".raw", ".psd"
        };

        public static HashSet<string> VideoExtensions { get; set; } = new HashSet<string>
        {
            ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg", ".3gp", ".ogv"
        };

        public static void CreateDirectoryIfNotExists(string path)
        {
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
                Console.WriteLine($"Directory created: {path}");
            }
            else
            {
                Console.WriteLine($"Directory already exists: {path}");
            }
        }

        public static void ForceSaveCheckpoint(string outputpath)
        {
            try
            {
                if(!File.Exists(Path.Combine(outputpath, "save")))
                {
                    File.WriteAllText(Path.Combine(outputpath, "save"), "111");
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Error saving file: {ex.Message}");
            }
        }

        public static void SaveToml(string toml, string path)
        {
            try
            {
                File.WriteAllText(path, toml);
            }
            catch (Exception ex)
            {
                throw new Exception($"Error saving file: {ex.Message}");
            }
        }

        public static Dictionary<string, List<string>?> GetAllFilesForDatasetConfiguration(string datasetPath)
        {
            var result = new Dictionary<string, List<string>?>();
            var datasetFolders = Directory.GetDirectories(datasetPath);

            var allExtensions = ImageExtensions.Union(VideoExtensions);

            foreach (var subdataset in datasetFolders)
            {
                var files = Directory
                    .EnumerateFiles(subdataset, "*.*", SearchOption.TopDirectoryOnly)
                    .Where(file => allExtensions.Any(ext => file.EndsWith(ext, StringComparison.OrdinalIgnoreCase))).ToList();

                string subdatasetName = Path.GetFileName(subdataset);
                result.Add(subdatasetName, files);
            }

            return result;
        }

        public static DatasetViewModel? GetDatasetConfiguration(string folder)
        {
            if (!Directory.Exists(folder))
            {
                Console.WriteLine($"Directory does not exist: {folder}");
                return null;
            }

            DirectoryInfo directoryInfo = new DirectoryInfo(folder);
            DateTime creationDate = directoryInfo.CreationTime;

            string datasetName = Path.GetFileName(folder);

            var datasetTomlFile = Directory.GetFiles(folder, "dataset.toml");
            var configurationTomlFile = Directory.GetFiles(folder, "config.toml");

            var datasetViewModel = new DatasetViewModel() {  Name = datasetName, Path = Path.Combine(folder, "dataset.toml"), FolderCreationDate = creationDate };

            if (datasetTomlFile.Length > 0)
            {
                var tomlContent = File.ReadAllText(datasetTomlFile[0]);
                var tomlTable = Toml.ToModel(tomlContent);
                datasetViewModel.DatasetToml = tomlTable;
            }

            if (configurationTomlFile.Length > 0)
            {
                var tomlContent = File.ReadAllText(configurationTomlFile[0]);
                var tomlTable = Toml.ToModel(tomlContent);

                datasetViewModel.ConfigurationTomlString = tomlContent;
                datasetViewModel.ConfigurationToml = tomlTable;
            }

            return datasetViewModel;
        }
        public static List<DatasetViewModel> GetDatasetConfigurations(string folderPath)
        {
            var datasetConfigurations = new List<DatasetViewModel>();

            try
            {
                if (!Directory.Exists(folderPath))
                {
                    Console.WriteLine($"Directory does not exist: {folderPath}");
                    return datasetConfigurations;
                }

                var datasetFolders = Directory.GetDirectories(folderPath);

                var key = 1;
                try
                {
                    foreach (var folder in datasetFolders)
                    {
                        var datasetConfiguration = GetDatasetConfiguration(folder);
                        datasetConfiguration.Key = key;
                        datasetConfigurations.Add(datasetConfiguration);
                        key++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing dataset files: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading directory {folderPath}: {ex.Message}");
            }

            return datasetConfigurations;
        }
    }
}
