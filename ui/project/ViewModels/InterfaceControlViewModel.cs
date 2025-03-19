namespace DiffusionPipeInterface.ViewModels
{
    public class InterfaceControlViewModel
    {
        public int TotalSteps { get; set; } = 0;
        public int StepsPerEpoch { get; set; } = 0;
        public int CurrentStep { get; set; } = 0;
        public int CurrentEpoch { get; set; } = 0;
        public int ProgressValueTrain { get; set; } = 0;
        public int ProgressValueUpload { get; set; } = 0;
        public bool CreateDatasetLocked { get; set; } = false;
        public bool DatasetSelectedLocked { get; set; } = true;
        public bool SavingDatasetLocked { get;set; } = false;
        public bool TrainingLocked { get; set; } = false;
        public bool IsUploading { get; set; } = false;
    }
}
