using DiffusionPipeInterface.Services;
using Microsoft.AspNetCore.Components.Server.Circuits;

namespace DiffusionPipeInterface.Hubs.Handler
{
    public class CustomCircuitHandler : CircuitHandler
    {
        private readonly ReconnectionService _reconnectionService;

        public CustomCircuitHandler(ReconnectionService reconnectionService)
        {
            _reconnectionService = reconnectionService;
        }

        public override Task OnConnectionUpAsync(Circuit circuit, CancellationToken cancellationToken)
        {
            _reconnectionService.RaiseReconnected();
            return base.OnConnectionUpAsync(circuit, cancellationToken);
        }
    }
}
