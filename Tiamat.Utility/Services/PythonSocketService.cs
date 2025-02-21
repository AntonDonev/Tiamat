using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Tiamat.Core.Services.Interfaces;

namespace Tiamat.Utility.Services
{
    public class PythonSocketService : BackgroundService
    {
        private readonly ILogger<PythonSocketService> _logger;
        private readonly string _serverAddress = "34.174.186.157";
        private readonly int _serverPort = 12346;
        private readonly IServiceScopeFactory _scopeFactory;
        private TcpClient _client;
        private NetworkStream _networkStream;
        private readonly Channel<string> _messageChannel = Channel.CreateUnbounded<string>();
        public bool IsConnected => _client != null && _client.Connected;

        public event EventHandler<string> MessageReceived;

        public PythonSocketService(ILogger<PythonSocketService> logger, IServiceScopeFactory scopeFactory)
        {
            _logger = logger;
            _scopeFactory = scopeFactory;
        }

        protected virtual void OnMessageReceived(string message)
        {
            MessageReceived?.Invoke(this, message);
        }

        public async Task EnqueueMessageAsync(string message)
        {
            await _messageChannel.Writer.WriteAsync(message);
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    using (_client = new TcpClient())
                    {
                        _logger.LogInformation("Connecting to Python server at {Address}:{Port}", _serverAddress, _serverPort);
                        await _client.ConnectAsync(_serverAddress, _serverPort);
                        _networkStream = _client.GetStream();

                        using (var scope = _scopeFactory.CreateScope())
                        {
                            var accountService = scope.ServiceProvider.GetRequiredService<IAccountService>();
                            foreach (var (id, ip) in accountService.AllAccounts())
                            {
                                var line = $"START {id} {ip}\n";
                                var data = Encoding.UTF8.GetBytes(line);
                                await _networkStream.WriteAsync(data, 0, data.Length, stoppingToken);
                                _logger.LogInformation("Sent account: {Line}", line.Trim());
                            }
                        }

                        var sendingTask = Task.Run(async () =>
                        {
                            while (await _messageChannel.Reader.WaitToReadAsync(stoppingToken))
                            {
                                while (_messageChannel.Reader.TryRead(out var outboundMessage))
                                {
                                    var fullMessage = outboundMessage.EndsWith("\n") ? outboundMessage : outboundMessage + "\n";
                                    var data = Encoding.UTF8.GetBytes(fullMessage);
                                    await _networkStream.WriteAsync(data, 0, data.Length, stoppingToken);
                                    _logger.LogInformation("Sent message: {Message}", outboundMessage.Trim());
                                }
                            }
                        }, stoppingToken);

                        byte[] buffer = new byte[1024];
                        while (!stoppingToken.IsCancellationRequested)
                        {
                            int bytesRead = await _networkStream.ReadAsync(buffer, 0, buffer.Length, stoppingToken);
                            if (bytesRead == 0)
                            {
                                _logger.LogWarning("Python server closed the connection.");
                                break;
                            }

                            string message = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                            OnMessageReceived(message);

                            string[] input = message.Split('|');
                            if (input[0] == "OPEN_CONFIRM")
                            {
                                try
                                {
                                    using (var scope = _scopeFactory.CreateScope())
                                    {
                                        var positionService = scope.ServiceProvider.GetRequiredService<IPositionService>();
                                        var accountService = scope.ServiceProvider.GetRequiredService<IAccountService>();

                                        string symbol = input[3].Trim();
                                        string type = input[2].Trim();
                                        string sizeStr = input[4].Trim();
                                        string riskStr = input[5].Trim();
                                        string openedAtStr = input[6].Trim();
                                        string fromIp = input[8].Replace("FROM_IP=", "").Trim();
                                        string id = input[7].Trim();

                                        var account = accountService.GetAccountByIp(fromIp);
                                        if (account == null)
                                        {
                                            _logger.LogError("Account with IP {FromIp} not found.", fromIp);
                                            continue; 
                                        }
                                        _logger.LogInformation("Received message: {Message}", message);
                                        _logger.LogInformation("Parsed: symbol={Symbol}, type={Type}, size={Size}, risk={Risk}, openedAt={OpenedAt}, fromIp={FromIp}, id={Id}",
                                            symbol, type, sizeStr, riskStr, openedAtStr, fromIp, id);
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

                                        string profit = input[2].Trim();
                                        string currentCapital = input[3].Trim();
                                        string closedAtStr = input[4].Trim();
                                        string id = input[6].Trim();
                                        string fromIp = input[7].Replace("FROM_IP=", "").Trim();

                                        positionService.ClosePosition(
                                            id,
                                            decimal.Parse(profit),
                                            decimal.Parse(currentCapital),
                                            DateTime.Parse(closedAtStr),
                                            fromIp);
                                    }
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogError(ex, "Error processing CLOSED_CONFIRM message");
                                }
                            }
                        }

                        await sendingTask;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error while connecting to Python server or during communication.");
                }

                await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
            }
        }
    }
}
