using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Tiamat.Utility
{
    public interface IPythonApiService
    {
            public bool IsConnected { get; }
            public bool ConnectionState { get; }

            Task<(bool IsSuccess, string ErrorMessage)> SendOpenCommandAsync(string id, string type, string symbol, string tp, string minutesUntilInvalid);
            Task<(bool IsSuccess, string ErrorMessage)> SendCloseCommandAsync(string id);
            Task<(bool IsSuccess, string ErrorMessage)> SendEditCommandAsync(string accountId, string maxRisk, string untradablePeriod);
            Task<(bool IsSuccess, string ErrorMessage)> StartAccountAsync(string accountId, string ip);
            Task<bool> IsPythonServiceHealthyAsync();
    }

        public class PythonApiService : IPythonApiService, IDisposable
        {
            private readonly ILogger<PythonApiService> _logger; // за логове, за грешки ако има...
            private readonly HttpClient _httpClient;
            private readonly string _pythonApiBaseUrl;
            private readonly string _apiKey;
            private DateTime _lastHealthCheck = DateTime.MinValue;
            private bool _isHealthy = false;
            private readonly object _healthLock = new object();

            public bool ConnectionState { get; private set; } = false;

            public PythonApiService(ILogger<PythonApiService> logger, IConfiguration configuration)
            {
                _logger = logger;
                _pythonApiBaseUrl = configuration["PythonApi:BaseUrl"] ?? "http://34.174.76.199:8000";
                _apiKey = configuration["PythonApi:ApiKey"] ?? "";

                _httpClient = new HttpClient();
                _httpClient.DefaultRequestHeaders.Add("x-api-key", _apiKey);
                _httpClient.Timeout = TimeSpan.FromSeconds(10);
            }

            public bool IsConnected
            {
                get
                {
                    if (DateTime.UtcNow - _lastHealthCheck > TimeSpan.FromMinutes(1))
                    {
                        CheckConnectionAsync().Wait();
                    }

                    return ConnectionState;
                }
            }

            private async Task CheckConnectionAsync()
            {
                try
                {
                    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                    var result = await _httpClient.GetAsync($"{_pythonApiBaseUrl}/health", timeoutCts.Token);

                    lock (_healthLock)
                    {
                        _isHealthy = result.IsSuccessStatusCode;
                        _lastHealthCheck = DateTime.UtcNow;
                        ConnectionState = _isHealthy;
                    }
                }
                catch (Exception)
                {
                    lock (_healthLock)
                    {
                        _isHealthy = false;
                        _lastHealthCheck = DateTime.UtcNow;
                        ConnectionState = false;
                    }
                }
            }

            public async Task<bool> IsPythonServiceHealthyAsync()
            {
                try
                {
                    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                    var response = await _httpClient.GetAsync($"{_pythonApiBaseUrl}/health", timeoutCts.Token);

                    bool isHealthy = response.IsSuccessStatusCode;

                    lock (_healthLock)
                    {
                        _isHealthy = isHealthy;
                        _lastHealthCheck = DateTime.UtcNow;
                        ConnectionState = isHealthy;
                    }

                    return isHealthy;
                }
                catch (OperationCanceledException)
                {
                    lock (_healthLock)
                    {
                        _isHealthy = false;
                        _lastHealthCheck = DateTime.UtcNow;
                        ConnectionState = false;
                    }
                    return false;
                }
                catch (Exception)
                {
                    lock (_healthLock)
                    {
                        _isHealthy = false;
                        _lastHealthCheck = DateTime.UtcNow;
                        ConnectionState = false;
                    }
                    return false;
                }
            }

            public async Task<(bool IsSuccess, string ErrorMessage)> SendOpenCommandAsync(string id, string type, string symbol, string tp, string minutesUntilInvalid)
            {
                try
                {
                    var message = $"OPEN|ID={id}|{type}|{symbol}|TP={tp}|MinutesUntilInvalid={minutesUntilInvalid}";

                    var content = new
                    {
                        command = "OPEN",
                        id = id,
                        type = type,
                        symbol = symbol,
                        tp = tp,
                        minutesUntilInvalid = minutesUntilInvalid
                    };

                    var jsonContent = new StringContent(JsonSerializer.Serialize(content), Encoding.UTF8, "application/json");
                    var response = await _httpClient.PostAsync($"{_pythonApiBaseUrl}/command", jsonContent);

                    if (response.IsSuccessStatusCode)
                    {
                        return (true, null);
                    }
                    else
                    {
                        var errorResponse = await response.Content.ReadAsStringAsync();

                        await CheckConnectionAsync();
                        return (false, $"Python API returned: {response.StatusCode} - {errorResponse}");
                    }
                }
                catch (Exception ex)
                {
                    await CheckConnectionAsync();
                    return (false, $"Error: {ex.Message}");
                }
            }

            public async Task<(bool IsSuccess, string ErrorMessage)> SendCloseCommandAsync(string id)
            {
                try
                {
                    var message = $"CLOSE|ID={id}";
                    var content = new
                    {
                        command = "CLOSE",
                        id = id
                    };

                    var jsonContent = new StringContent(JsonSerializer.Serialize(content), Encoding.UTF8, "application/json");
                    var response = await _httpClient.PostAsync($"{_pythonApiBaseUrl}/command", jsonContent);

                    if (response.IsSuccessStatusCode)
                    {
                        return (true, null);
                    }
                    else
                    {
                        var errorResponse = await response.Content.ReadAsStringAsync();

                        await CheckConnectionAsync();
                        return (false, $"Python API returned: {response.StatusCode} - {errorResponse}");
                    }
                }
                catch (Exception ex)
                {
                    await CheckConnectionAsync();
                    return (false, $"Error: {ex.Message}");
                }
            }

            public async Task<(bool IsSuccess, string ErrorMessage)> SendEditCommandAsync(string accountId, string maxRisk, string untradablePeriod)
            {
                try
                {
                    var message = $"EDIT|{accountId}|{maxRisk}|{untradablePeriod}";
                    var content = new
                    {
                        command = "EDIT",
                        accountId = accountId,
                        maxRisk = maxRisk,
                        untradablePeriod = untradablePeriod
                    };

                    var jsonContent = new StringContent(JsonSerializer.Serialize(content), Encoding.UTF8, "application/json");
                    var response = await _httpClient.PostAsync($"{_pythonApiBaseUrl}/command", jsonContent);

                    if (response.IsSuccessStatusCode)
                    {
                        return (true, null);
                    }
                    else
                    {
                        var errorResponse = await response.Content.ReadAsStringAsync();

                        await CheckConnectionAsync();
                        return (false, $"Python API returned: {response.StatusCode} - {errorResponse}");
                    }
                }
                catch (Exception ex)
                {
                    await CheckConnectionAsync();
                    return (false, $"Error: {ex.Message}");
                }
            }

            public async Task<(bool IsSuccess, string ErrorMessage)> StartAccountAsync(string accountId, string ip)
            {
                try
                {
                    var message = $"START {accountId} {ip}";
                    var content = new
                    {
                        command = "START",
                        accountId = accountId,
                        ip = ip
                    };

                    var jsonContent = new StringContent(JsonSerializer.Serialize(content), Encoding.UTF8, "application/json");
                    var response = await _httpClient.PostAsync($"{_pythonApiBaseUrl}/command", jsonContent);

                    if (response.IsSuccessStatusCode)
                    {
                        return (true, null);
                    }
                    else
                    {
                        var errorResponse = await response.Content.ReadAsStringAsync();

                        await CheckConnectionAsync();
                        return (false, $"Python API returned: {response.StatusCode} - {errorResponse}");
                    }
                }
                catch (Exception ex)
                {
                    await CheckConnectionAsync();
                    return (false, $"Error: {ex.Message}");
                }
            }

            public void Dispose()
            {
                _httpClient?.Dispose();
            }
        }
}
