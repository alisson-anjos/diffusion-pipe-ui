using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DiffusionPipeInterface.Migrations
{
    /// <inheritdoc />
    public partial class Initial : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "AdapterConfigurations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Type = table.Column<string>(type: "TEXT", nullable: false),
                    Rank = table.Column<int>(type: "INTEGER", nullable: false),
                    Dtype = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AdapterConfigurations", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "DatasetConfigurations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Resolutions = table.Column<string>(type: "TEXT", nullable: false),
                    EnableARBucket = table.Column<bool>(type: "INTEGER", nullable: false),
                    MinAR = table.Column<float>(type: "REAL", nullable: false),
                    MaxAR = table.Column<float>(type: "REAL", nullable: false),
                    NumARBuckets = table.Column<int>(type: "INTEGER", nullable: false),
                    ARBuckets = table.Column<string>(type: "TEXT", nullable: true),
                    FrameBuckets = table.Column<string>(type: "TEXT", nullable: false),
                    NumRepeats = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_DatasetConfigurations", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "ModelConfigurations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Type = table.Column<int>(type: "INTEGER", nullable: false),
                    CheckpointPath = table.Column<string>(type: "TEXT", nullable: false),
                    DiffusersPath = table.Column<string>(type: "TEXT", nullable: false),
                    TransformerPath = table.Column<string>(type: "TEXT", nullable: false),
                    VaePath = table.Column<string>(type: "TEXT", nullable: false),
                    TextEncoderPath = table.Column<string>(type: "TEXT", nullable: false),
                    LlmPath = table.Column<string>(type: "TEXT", nullable: false),
                    ClipPath = table.Column<string>(type: "TEXT", nullable: false),
                    Dtype = table.Column<int>(type: "INTEGER", nullable: false),
                    TransformerDType = table.Column<int>(type: "INTEGER", nullable: false),
                    TimestepSampleMethod = table.Column<int>(type: "INTEGER", nullable: false),
                    ModelType = table.Column<string>(type: "TEXT", maxLength: 21, nullable: false),
                    FluxShift = table.Column<bool>(type: "INTEGER", nullable: true),
                    FluxModelConfiguration_FluxShift = table.Column<bool>(type: "INTEGER", nullable: true),
                    BypassGuidanceEmbedding = table.Column<bool>(type: "INTEGER", nullable: true),
                    CkptPath = table.Column<string>(type: "TEXT", nullable: true),
                    SingleFilePath = table.Column<string>(type: "TEXT", nullable: true),
                    LuminaShift = table.Column<bool>(type: "INTEGER", nullable: true),
                    UnetLr = table.Column<float>(type: "REAL", nullable: true),
                    TextEncoder1Lr = table.Column<float>(type: "REAL", nullable: true),
                    TextEncoder2Lr = table.Column<float>(type: "REAL", nullable: true),
                    VPred = table.Column<bool>(type: "INTEGER", nullable: true),
                    MinSnrGamma = table.Column<float>(type: "REAL", nullable: true),
                    DebiasedEstimationLoss = table.Column<bool>(type: "INTEGER", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ModelConfigurations", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "OptimizerConfigurations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Type = table.Column<int>(type: "INTEGER", nullable: false),
                    Lr = table.Column<double>(type: "REAL", nullable: false),
                    WeightDecay = table.Column<double>(type: "REAL", nullable: false),
                    Eps = table.Column<double>(type: "REAL", nullable: false),
                    Betas = table.Column<string>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_OptimizerConfigurations", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "DirectoryConfigurations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Path = table.Column<string>(type: "TEXT", nullable: false),
                    MaskPath = table.Column<string>(type: "TEXT", nullable: true),
                    NumRepeats = table.Column<int>(type: "INTEGER", nullable: false),
                    ARBuckets = table.Column<string>(type: "TEXT", nullable: true),
                    Resolutions = table.Column<string>(type: "TEXT", nullable: true),
                    FrameBuckets = table.Column<string>(type: "TEXT", nullable: true),
                    DatasetConfigurationId = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_DirectoryConfigurations", x => x.Id);
                    table.ForeignKey(
                        name: "FK_DirectoryConfigurations_DatasetConfigurations_DatasetConfigurationId",
                        column: x => x.DatasetConfigurationId,
                        principalTable: "DatasetConfigurations",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "TrainConfigurations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    OutputDir = table.Column<string>(type: "TEXT", nullable: false),
                    DatasetConfig = table.Column<string>(type: "TEXT", nullable: false),
                    Epochs = table.Column<int>(type: "INTEGER", nullable: false),
                    MicroBatchSizePerGPU = table.Column<int>(type: "INTEGER", nullable: false),
                    PipelineStages = table.Column<int>(type: "INTEGER", nullable: false),
                    GradientAccumulationSteps = table.Column<int>(type: "INTEGER", nullable: false),
                    GradientClipping = table.Column<float>(type: "REAL", nullable: false),
                    WarmupSteps = table.Column<int>(type: "INTEGER", nullable: false),
                    EvalEveryNEpochs = table.Column<int>(type: "INTEGER", nullable: false),
                    EvalBeforeFirstSteps = table.Column<bool>(type: "INTEGER", nullable: false),
                    EvalMicroBatchSizePerGPU = table.Column<int>(type: "INTEGER", nullable: false),
                    EvalGradientAccumulationSteps = table.Column<int>(type: "INTEGER", nullable: false),
                    SaveEveryNEpochs = table.Column<int>(type: "INTEGER", nullable: false),
                    CheckpointEveryNMinutes = table.Column<int>(type: "INTEGER", nullable: false),
                    ActivationCheckpointing = table.Column<bool>(type: "INTEGER", nullable: false),
                    PartitionMethod = table.Column<int>(type: "INTEGER", nullable: false),
                    SaveDType = table.Column<int>(type: "INTEGER", nullable: false),
                    CachingBatchSize = table.Column<int>(type: "INTEGER", nullable: false),
                    StepsPerPrint = table.Column<int>(type: "INTEGER", nullable: false),
                    VideoClipMode = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TrainConfigurations", x => x.Id);
                    table.ForeignKey(
                        name: "FK_TrainConfigurations_AdapterConfigurations_Id",
                        column: x => x.Id,
                        principalTable: "AdapterConfigurations",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_TrainConfigurations_ModelConfigurations_Id",
                        column: x => x.Id,
                        principalTable: "ModelConfigurations",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_TrainConfigurations_OptimizerConfigurations_Id",
                        column: x => x.Id,
                        principalTable: "OptimizerConfigurations",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_DirectoryConfigurations_DatasetConfigurationId",
                table: "DirectoryConfigurations",
                column: "DatasetConfigurationId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "DirectoryConfigurations");

            migrationBuilder.DropTable(
                name: "TrainConfigurations");

            migrationBuilder.DropTable(
                name: "DatasetConfigurations");

            migrationBuilder.DropTable(
                name: "AdapterConfigurations");

            migrationBuilder.DropTable(
                name: "ModelConfigurations");

            migrationBuilder.DropTable(
                name: "OptimizerConfigurations");
        }
    }
}
