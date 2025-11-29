using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Tiamat.Core.Services.Interfaces;
using Tiamat.Models;
using Tiamat.Utility;

namespace Tiamat.Utility.Services
{
    public class PythonInitializationService : BackgroundService
    {
        private readonly ILogger<PythonInitializationService> _logger;
        private readonly IPythonApiService _pythonApiService;
        private readonly IServiceScopeFactory _serviceScopeFactory;
        private bool _isInitialized = false;
        private readonly TimeSpan _retryInterval = TimeSpan.FromSeconds(10);

        public PythonInitializationService(
            ILogger<PythonInitializationService> logger,
            IPythonApiService pythonApiService,
            IServiceScopeFactory serviceScopeFactory)
        {
            _logger = logger;
            _pythonApiService = pythonApiService;
            _serviceScopeFactory = serviceScopeFactory;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isInitialized && await _pythonApiService.IsPythonServiceHealthyAsync())
                    {
                        await InitializeAllActiveAccountsAsync();
                        _isInitialized = true;
                    }
                    else if (!_pythonApiService.ConnectionState)
                    {
                        if (_isInitialized)
                        {
                            _isInitialized = false;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _isInitialized = false;
                }

                await Task.Delay(_retryInterval, stoppingToken);
            }
        }

        private async Task InitializeAllActiveAccountsAsync()
        {

            try
            {
                using (var scope = _serviceScopeFactory.CreateScope())
                {
                    var accountService = scope.ServiceProvider.GetRequiredService<IAccountService>();

                    var allAccounts = await accountService.GetAllAccountsAsync();
                    var activeAccounts = allAccounts.Where(a => a.Status == AccountStatus.Active).ToList();

                    _logger.LogInformation($"Found {activeAccounts.Count} active accounts to initialize");

                    foreach (var account in activeAccounts)
                    {
                        if (!string.IsNullOrEmpty(account.Affiliated_HWID))
                        {
                            _logger.LogInformation($"Sending START command for account {account.Id} with HWID {account.Affiliated_HWID}");

                            var result = await _pythonApiService.StartAccountAsync(account.Id.ToString(), account.Affiliated_HWID);

                            if (result.IsSuccess)
                            {
                                _logger.LogInformation($"Successfully started account {account.Id}");
                            }
                            else
                            {
                                _logger.LogWarning($"Failed to start account {account.Id}: {result.ErrorMessage}");
                            }

                            await Task.Delay(200);
                        }
                        else
                        {
                            _logger.LogWarning($"Account {account.Id} has no affiliated HWID, skipping");
                        }
                    }
                }

                _logger.LogInformation("Completed initialization of all active accounts");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize accounts");
                _isInitialized = false;
            }
        }
    }
}