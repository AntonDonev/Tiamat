using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Utility.Services
{
    public class PythonSocketService : BackgroundService
    {
        private readonly ILogger<PythonSocketService> _logger;
        private readonly string _serverAddress;
        private readonly int _serverPort;

        public PythonSocketService(ILogger<PythonSocketService> logger)
        {
            _logger = logger;
            _serverAddress = "127.0.0.1";
            _serverPort = 12346;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    using (var client = new TcpClient())
                    {
                        _logger.LogInformation("Connecting to Python server at {Address}:{Port}", _serverAddress, _serverPort);
                        await client.ConnectAsync(_serverAddress, _serverPort);

                        using (var networkStream = client.GetStream())
                        {
                            byte[] buffer = new byte[1024];
                            while (!stoppingToken.IsCancellationRequested)
                            {
                                int bytesRead = await networkStream.ReadAsync(buffer, 0, buffer.Length, stoppingToken);
                                if (bytesRead == 0)
                                {
                                    _logger.LogWarning("Python server closed the connection.");
                                    break;
                                }

                                string message = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                                _logger.LogInformation("Received from Python server: {Message}", message);

                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error while connecting to Python server.");
                }

                await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
            }
        }
    }
}