using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.Models;
using DiffusionPipeInterface.Models.Models;
using DiffusionPipeInterface.Utils;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

namespace DiffusionPipeInterface.Database
{
    public class AppDbContext : DbContext
    {
        protected readonly IConfiguration Configuration;

        public AppDbContext(IConfiguration configuration)
        {
            Configuration = configuration;
        }
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlite(Configuration.GetConnectionString("app"));
        }

        public DbSet<TrainConfiguration> TrainConfigurations { get; set; }
        public DbSet<DatasetConfiguration> DatasetConfigurations { get; set; }
        public DbSet<DatasetDirectoryConfiguration> DirectoryConfigurations { get; set; }
        public DbSet<AdapterConfiguration> AdapterConfigurations { get; set; }
        public DbSet<ModelConfiguration> ModelConfigurations { get; set; }
        public DbSet<OptimizerConfiguration> OptimizerConfigurations { get; set; }
       

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<ModelConfiguration>()
                .HasDiscriminator<string>("ModelType")
                .HasValue<SdxlModelConfiguration>(ModelType.SDXL.GetDescription())
                .HasValue<FluxModelConfiguration>(ModelType.Flux.GetDescription())
                .HasValue<LtxModelConfiguration>(ModelType.LTX.GetDescription())
                .HasValue<HunyuanModelConfiguration>(ModelType.Hunyuan.GetDescription())
                .HasValue<CosmosModelConfiguration>(ModelType.Cosmos.GetDescription())
                .HasValue<LuminaModelConfiguration>(ModelType.Lumina.GetDescription())
                .HasValue<WanModelConfiguration>(ModelType.Wan21.GetDescription())
                .HasValue<ChromaModelConfiguration>(ModelType.Chroma.GetDescription());

            modelBuilder.Entity<DatasetConfiguration>()
                .HasMany(d => d.Directories)
                .WithOne(d => d.DatasetConfiguration)
                .HasForeignKey(d => d.DatasetConfigurationId);

            modelBuilder.Entity<TrainConfiguration>()
                .HasOne(t => t.ModelConfiguration)
                .WithOne()
                .HasForeignKey<TrainConfiguration>(t => t.Id);

            modelBuilder.Entity<TrainConfiguration>()
                .HasOne(t => t.AdapterConfiguration)
                .WithOne()
                .HasForeignKey<TrainConfiguration>(t => t.Id);

            modelBuilder.Entity<TrainConfiguration>()
                .HasOne(t => t.OptimizerConfiguration)
                .WithOne()
                .HasForeignKey<TrainConfiguration>(t => t.Id);
        }
    }
}
