namespace DiffusionPipeInterface.Services
{
    public class ReconnectionService
    {
        public event Action? Reconnected;

        public void RaiseReconnected() => Reconnected?.Invoke();
    }
}
