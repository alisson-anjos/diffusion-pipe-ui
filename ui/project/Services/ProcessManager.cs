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

        public Guid StartProcess(string command, string workingDirectory, string datasetName)
        {
            var processId = Guid.NewGuid();
            var processInfo = new ProcessInfo
            {
                Id = processId,
                Command = command,
                WorkingDirectory = workingDirectory,
                DatasetName = datasetName,
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
                        var logLine = args.Data;


                        if (logLine.Contains("Total steps:"))
                        {
                            var match = System.Text.RegularExpressions.Regex.Match(logLine, @"Total steps:\s*(\d+)");
                            if (match.Success)
                            {
                                processInfo.TotalSteps = int.Parse(match.Groups[1].Value);
                            }
                        }

                        if (logLine.Contains("step="))
                        {
                            var match = System.Text.RegularExpressions.Regex.Match(logLine, @"step=(\d+)");
                            if (match.Success)
                            {
                                processInfo.CurrentStep = int.Parse(match.Groups[1].Value);
                            }
                        }

                        if (logLine.Contains("Steps per epoch:"))
                        {
                            var match = System.Text.RegularExpressions.Regex.Match(logLine, @"Steps per epoch:\s*(\d+)");
                            if (match.Success)
                            {
                                processInfo.StepsPerEpoch = int.Parse(match.Groups[1].Value);
                            }
                        }

                        if (logLine.Contains("Started new epoch:"))
                        {
                            var match = System.Text.RegularExpressions.Regex.Match(logLine, @"Started new epoch:\s*(\d+)");
                            if (match.Success)
                            {
                                processInfo.CurrentEpoch = int.Parse(match.Groups[1].Value);
                            }
                        }

                        processInfo.Logs.Enqueue(logLine);
                        LogReceived?.Invoke(this, new ProcessLogEventArgs(processInfo.Id, logLine));
                    }
                };

                process.ErrorDataReceived += (sender, args) =>
                {
                    if (!string.IsNullOrEmpty(args.Data))
                    {
                        string logLine = args.Data;
                        processInfo.Logs.Enqueue(logLine);
                        LogReceived?.Invoke(this, new ProcessLogEventArgs(processInfo.Id, logLine));
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
