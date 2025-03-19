using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Text.Json;
using DiffusionPipeInterface.Enums;

namespace DiffusionPipeInterface.Models
{
    public class DatasetConfiguration
    {
        [Key]
        public int Id { get; set; }

        public string Name { get; set; } = null!;

        public ModelType Model { get; set; }

        private string _resolutionsJson;

        [NotMapped]
        public int[] Resolutions
        {
            get => JsonSerializer.Deserialize<int[]>(_resolutionsJson) ?? Array.Empty<int>();
            set => _resolutionsJson = JsonSerializer.Serialize(value);
        }

        [Column("Resolutions")]
        public string ResolutionsJson
        {
            get => _resolutionsJson;
            set => _resolutionsJson = value;
        }

        public bool EnableARBucket { get; set; } = true;
        public float MinAR { get; set; } = 0.5f;
        public float MaxAR { get; set; } = 2.0f;
        public int NumARBuckets { get; set; } = 7;

        private string? _arBucketsJson;

        [NotMapped]
        public float[]? ARBuckets
        {
            get => _arBucketsJson != null && !string.IsNullOrEmpty(_arBucketsJson) ? JsonSerializer.Deserialize<float[]>(_arBucketsJson) : null;
            set => _arBucketsJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [Column("ARBuckets")]
        public string? ARBucketsJson
        {
            get => _arBucketsJson;
            set => _arBucketsJson = value;
        }

        private string _frameBucketsJson;

        [NotMapped]
        public int[] FrameBuckets
        {
            get => JsonSerializer.Deserialize<int[]>(_frameBucketsJson) ?? Array.Empty<int>();
            set => _frameBucketsJson = JsonSerializer.Serialize(value);
        }

        [Column("FrameBuckets")]
        public string FrameBucketsJson
        {
            get => _frameBucketsJson;
            set => _frameBucketsJson = value;
        }

        public int NumRepeats { get; set; } = 10;

        public virtual List<DatasetDirectoryConfiguration>? Directories { get; set; }

    }
}
