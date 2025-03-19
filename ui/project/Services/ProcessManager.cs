using DiffusionPipeInterface.Models.Process;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace DiffusionPipeInterface.Services
{
    public class ProcessManager
    {
        private static readonly ConcurrentDictionary<Guid, ProcessInfo> Processes = new();

        public event EventHandler<ProcessLogEventArgs> LogReceived;

        public event EventHandler<Guid> ProcessCompleted;

        public Guid StartProcess(string command, string workingDirectory)
        {
            var processId = Guid.NewGuid();
            var processInfo = new ProcessInfo
            {
                Id = processId,
                Command = command,
                WorkingDirectory = workingDirectory,
                Logs = new System.Collections.Concurrent.ConcurrentQueue<string>(),
                IsRunning = true
            };

            Processes[processId] = processInfo;

            Task.Run(() => RunProcessAsync(processInfo));

            return processId;
        }

        private async Task RunProcessAsync(ProcessInfo processInfo)
        {
            var processStartInfo = new ProcessStartInfo
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = processInfo.WorkingDirectory
            };

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                processStartInfo.FileName = "cmd.exe";
                processStartInfo.Arguments = $"/c \"{processInfo.Command}\"";
            }
            else
            {
                processStartInfo.FileName = "/bin/bash";
                processStartInfo.Arguments = $"-c \"{processInfo.Command}\"";
            }

            using (var process = new Process { StartInfo = processStartInfo })
            {
                processInfo.Process = process;

                process.OutputDataReceived += (sender, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        processInfo.Logs.Enqueue(args.Data);
                        LogReceived?.Invoke(this, new ProcessLogEventArgs(processInfo.Id, args.Data));
                    }
                };

                process.ErrorDataReceived += (sender, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        processInfo.Logs.Enqueue($"{args.Data}");
                        LogReceived?.Invoke(this, new ProcessLogEventArgs(processInfo.Id, $"{args.Data}"));
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await Task.Run(() => process.WaitForExit());

                processInfo.IsRunning = false;
                processInfo.ExitCode = process.ExitCode;

                ProcessCompleted?.Invoke(this, processInfo.Id);
            }
        }

        public void StopProcess(Guid processId)
        {
            if (Processes.TryGetValue(processId, out var processInfo))
            {
                try
                {
                    if (processInfo.Process != null && !processInfo.Process.HasExited)
                    {
                        processInfo.Process.Kill(true);
                    }

                    processInfo.IsRunning = false;
                    processInfo.ExitCode = -1;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error to stop process: {ex.Message}");
                }
            }
        }

        public ProcessInfo GetProcessInfo(Guid processId)
        {
            return Processes.TryGetValue(processId, out var processInfo) ? processInfo : null;
        }
    }
}
