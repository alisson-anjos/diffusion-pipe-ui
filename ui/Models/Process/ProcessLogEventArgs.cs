namespace DiffusionPipeInterface.Models.Process
{
    public class ProcessLogEventArgs : EventArgs
    {
        public Guid ProcessId { get; }
        public string Log { get; }

        public ProcessLogEventArgs(Guid processId, string log)
        {
            ProcessId = processId;
            Log = log;
        }
    }
}
