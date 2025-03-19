﻿using System.Diagnostics;

namespace DiffusionPipeInterface.Models.Process
{
    public class ProcessInfo
    {
        public Guid Id { get; set; }
        public string Command { get; set; }
        public string WorkingDirectory { get; set; }
        public System.Collections.Concurrent.ConcurrentQueue<string> Logs { get; set; }
        public bool IsRunning { get; set; }
        public int? ExitCode { get; set; }
        public System.Diagnostics.Process Process { get; set; }
    }
}
