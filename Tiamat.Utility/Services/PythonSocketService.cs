using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Tiamat.Core.Services.Interfaces;

namespace Tiamat.Utility.Services
{
    public class PythonSocketService : BackgroundService
    {
        private readonly ILogger<PythonSocketService> _logger;
        private readonly string _serverAddress;
        private readonly int _serverPort;
        private readonly IServiceScopeFactory _scopeFactory;

        public event EventHandler<string> MessageReceived;

        public PythonSocketService(ILogger<PythonSocketService> logger,
                                   IServiceScopeFactory scopeFactory)
        {
            _logger = logger;
            _scopeFactory = scopeFactory;
            _serverAddress = "34.174.186.157";
            _serverPort = 12346;
        }

        protected virtual void OnMessageReceived(string message)
        {
            MessageReceived?.Invoke(this, message);
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
                            // Send account data using a scope.
                            using (var scope = _scopeFactory.CreateScope())
                            {
                                var accountService = scope.ServiceProvider.GetRequiredService<IAccountService>();
                                foreach (var (id, ip) in accountService.AllAccounts())
                                {
                                    var line = $"START {id} {ip}\n";
                                    var data = Encoding.UTF8.GetBytes(line);
                                    await networkStream.WriteAsync(data, 0, data.Length, stoppingToken);
                                    _logger.LogInformation("Sent account: {Line}", line.Trim());
                                }
                            }

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
                                string[] input = message.Split('|');

                                if (input[0] == "OPEN_CONFIRM")
                                {
                                    try
                                    {
                                        // Resolve services in a new scope.
                                        using (var scope = _scopeFactory.CreateScope())
                                        {
                                            var positionService = scope.ServiceProvider.GetRequiredService<IPositionService>();
                                            var accountService = scope.ServiceProvider.GetRequiredService<IAccountService>();

                                            // Remove key names before parsing.
                                            string symbol = input[3].Trim();
                                            string type = input[2].Trim();
                                            string sizeStr = input[4].Trim();
                                            string riskStr = input[5].Trim();
                                            string openedAtStr = input[6].Trim();
                                            string fromIp = input[8].Replace("FROM_IP =", "").Trim();
                                            string id = input[7].Trim();

                                            var account = accountService.GetAccountByIp(fromIp);
                                            if (account == null)
                                            {
                                                _logger.LogError("Account with IP {FromIp} not found.", fromIp);
                                                continue; // Skip processing if account not found
                                            }

                                            // Pass the entire account object to CreatePosition.
                                            positionService.CreatePosition(
                                                symbol,
                                                type,
                                                account,
                                                decimal.Parse(sizeStr),
                                                decimal.Parse(riskStr),
                                                DateTime.Parse(openedAtStr),
                                                id);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        _logger.LogError(ex, "Error processing OPEN_CONFIRM message");
                                    }
                                }
                                else if (input[0] == "CLOSED_CONFIRM")
                                {
                                    try
                                    {
                                        using (var scope = _scopeFactory.CreateScope())
                                        {
                                            var positionService = scope.ServiceProvider.GetRequiredService<IPositionService>();

                                            // Remove key names before parsing.
                                            string profit = input[2].Trim();
                                            string currentCapital = input[3].Trim();
                                            string closedAtStr = input[4].Trim();
                                            string id = input[5].Trim();

                                            positionService.ClosePosition(
                                                id,
                                                decimal.Parse(profit),
                                                decimal.Parse(currentCapital),
                                                DateTime.Parse(closedAtStr));
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        _logger.LogError(ex, "Error processing CLOSED_CONFIRM message");
                                    }
                                }

                                OnMessageReceived(message);
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

