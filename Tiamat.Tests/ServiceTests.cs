using NUnit.Framework;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Moq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Tiamat.Core.Services;
using Tiamat.DataAccess;
using Tiamat.Models;
using Tiamat.Core.Services.Interfaces;

namespace Tiamat.Tests
{
    [TestFixture]
    public class ServiceTests
    {
        private TiamatDbContext _context;
        private Mock<ILogger<PositionService>> _mockLogger;

        [SetUp]
        public void Setup()
        {
            var options = new DbContextOptionsBuilder<TiamatDbContext>()
                .UseInMemoryDatabase(databaseName: Guid.NewGuid().ToString())
                .Options;

            _context = new TiamatDbContext(options);
            _mockLogger = new Mock<ILogger<PositionService>>();
        }

        [TearDown]
        public void TearDown()
        {
            _context.Database.EnsureDeleted();
            _context.Dispose();
        }

        [Test]
        public async Task GetAllAccountsAsync_ReturnsAllAccounts()
        {
            var service = new AccountService(_context);
            var setting = new AccountSetting { AccountSettingId = Guid.NewGuid(), SettingName = "Test Setting", MaxRiskPerTrade = 10, UntradablePeriodMinutes = 10 };
            await _context.AccountSettings.AddAsync(setting);

            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    AccountName = "Account 1",
                    AccountSettingsId = setting.AccountSettingId,
                    Platform = "MT4",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    Status = AccountStatus.Active,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    AccountName = "Account 2",
                    AccountSettingsId = setting.AccountSettingId,
                    Platform = "MT5",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    Status = AccountStatus.Active,
                    UserId = Guid.NewGuid()
                }
            };
            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.GetAllAccountsAsync();

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.All(a => a.AccountSetting != null), Is.True);
        }

        [Test]
        public async Task GetAccountByIdAsync_ExistingId_ReturnsAccount()
        {
            var service = new AccountService(_context);
            var accountId = Guid.NewGuid();
            var settingId = Guid.NewGuid();
            var setting = new AccountSetting
            {
                AccountSettingId = settingId,
                SettingName = "Test Setting",
                MaxRiskPerTrade = 10,
                UntradablePeriodMinutes = 10
            };
            await _context.AccountSettings.AddAsync(setting);

            var account = new Account
            {
                Id = accountId,
                AccountName = "Test Account",
                AccountSettingsId = settingId,
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };
            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();

            var result = await service.GetAccountByIdAsync(accountId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Id, Is.EqualTo(accountId));
            Assert.That(result.AccountSetting, Is.Not.Null);
            Assert.That(result.AccountSetting.AccountSettingId, Is.EqualTo(settingId));
        }

        [Test]
        public async Task GetAccountByIdAsync_NonExistingId_ReturnsNull()
        {
            var service = new AccountService(_context);
            var result = await service.GetAccountByIdAsync(Guid.NewGuid());
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task GetAccountByIpAsync_ExistingIp_ReturnsAccount()
        {
            var service = new AccountService(_context);
            var ip = "192.168.1.1";
            var account = new Account
            {
                Id = Guid.NewGuid(),
                AccountName = "Test Account",
                Affiliated_IP = ip,
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };
            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();

            var result = await service.GetAccountByIpAsync(ip);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Affiliated_IP, Is.EqualTo(ip));
        }

        [Test]
        public async Task GetAccountByIpAsync_NonExistingIp_ReturnsNull()
        {
            var service = new AccountService(_context);
            var result = await service.GetAccountByIpAsync("192.168.1.123");
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task CreateAccountAsync_AddsNewAccount()
        {
            var service = new AccountService(_context);
            var accountId = Guid.NewGuid();
            var account = new Account
            {
                Id = accountId,
                AccountName = "New Account",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };

            await service.CreateAccountAsync(account);

            var result = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.AccountName, Is.EqualTo("New Account"));
        }

        [Test]
        public async Task UpdateAccountAsync_UpdatesExistingAccount()
        {
            var service = new AccountService(_context);
            var accountId = Guid.NewGuid();
            var account = new Account
            {
                Id = accountId,
                AccountName = "Original Name",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };
            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();

            account.AccountName = "Updated Name";
            await service.UpdateAccountAsync(account);

            var result = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.AccountName, Is.EqualTo("Updated Name"));
        }

        [Test]
        public async Task DeleteAccountAsync_ExistingAccount_DeletesAccount()
        {
            var service = new AccountService(_context);
            var accountId = Guid.NewGuid();
            var account = new Account
            {
                Id = accountId,
                AccountName = "Test Account",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };
            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();

            await service.DeleteAccountAsync(accountId);

            var result = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task DeleteAccountAsync_NonExistingAccount_DoesNothing()
        {
            var service = new AccountService(_context);
            var initialCount = await _context.Accounts.CountAsync();

            await service.DeleteAccountAsync(Guid.NewGuid());

            var finalCount = await _context.Accounts.CountAsync();
            Assert.That(finalCount, Is.EqualTo(initialCount));
        }

        [Test]
        public async Task AccountReviewAsync_WithFiveParams_AccountExists_UpdatesAccount()
        {
            var service = new AccountService(_context);
            var accountId = Guid.NewGuid();
            var account = new Account
            {
                Id = accountId,
                Status = AccountStatus.Pending,
                AccountName = "Test Account",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                UserId = Guid.NewGuid()
            };
            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();

            await service.AccountReviewAsync(AccountStatus.Active, accountId, "VPS1", "admin@example.com", "192.168.1.1");

            var result = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Status, Is.EqualTo(AccountStatus.Active));
            Assert.That(result.VPSName, Is.EqualTo("VPS1"));
            Assert.That(result.AdminEmail, Is.EqualTo("admin@example.com"));
            Assert.That(result.Affiliated_IP, Is.EqualTo("192.168.1.1"));
        }

        [Test]
        public async Task AccountReviewAsync_WithFiveParams_AccountDoesNotExist_DoesNothing()
        {
            var service = new AccountService(_context);

            await service.AccountReviewAsync(AccountStatus.Active, Guid.NewGuid(), "VPS1", "admin@example.com", "192.168.1.1");

            var accounts = await _context.Accounts.ToListAsync();
            Assert.That(accounts, Is.Empty);
        }

        [Test]
        public async Task AccountReviewAsync_WithTwoParams_AccountExists_UpdatesStatus()
        {
            var service = new AccountService(_context);
            var accountId = Guid.NewGuid();
            var account = new Account
            {
                Id = accountId,
                Status = AccountStatus.Pending,
                AccountName = "Test Account",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                UserId = Guid.NewGuid()
            };
            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();

            await service.AccountReviewAsync(AccountStatus.Active, accountId);

            var result = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Status, Is.EqualTo(AccountStatus.Active));
        }

        [Test]
        public async Task AccountReviewAsync_WithTwoParams_AccountDoesNotExist_DoesNothing()
        {
            var service = new AccountService(_context);

            await service.AccountReviewAsync(AccountStatus.Active, Guid.NewGuid());

            var accounts = await _context.Accounts.ToListAsync();
            Assert.That(accounts, Is.Empty);
        }

        [Test]
        public async Task FilterAccountsAsync_WithAllFilters_ReturnsFilteredAccounts()
        {
            var service = new AccountService(_context);
            var settingId1 = Guid.NewGuid();
            var settingId2 = Guid.NewGuid();

            var setting1 = new AccountSetting { AccountSettingId = settingId1, SettingName = "Setting 1", MaxRiskPerTrade = 10, UntradablePeriodMinutes = 10 };
            var setting2 = new AccountSetting { AccountSettingId = settingId2, SettingName = "Setting 2", MaxRiskPerTrade = 20, UntradablePeriodMinutes = 20 };
            await _context.AccountSettings.AddRangeAsync(new[] { setting1, setting2 });

            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId1,
                    AccountName = "Account 1",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Failed,
                    AccountSettingsId = settingId1,
                    AccountName = "Account 2",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT5",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId2,
                    AccountName = "Account 3",
                    BrokerLogin = "login3",
                    BrokerPassword = "pass3",
                    BrokerServer = "server3",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 3000,
                    HighestCapital = 3000,
                    InitialCapital = 3000,
                    LowestCapital = 3000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT5",
                    Status = AccountStatus.Pending,
                    AccountSettingsId = settingId2,
                    AccountName = "Account 4",
                    BrokerLogin = "login4",
                    BrokerPassword = "pass4",
                    BrokerServer = "server4",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 4000,
                    HighestCapital = 4000,
                    InitialCapital = 4000,
                    LowestCapital = 4000,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.FilterAccountsAsync("MT4", AccountStatus.Active, settingId1);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(1));
            Assert.That(result.First().Platform, Is.EqualTo("MT4"));
            Assert.That(result.First().Status, Is.EqualTo(AccountStatus.Active));
            Assert.That(result.First().AccountSettingsId, Is.EqualTo(settingId1));
        }

        [Test]
        public async Task FilterAccountsAsync_WithNoPlatformFilter_FiltersOnlyByStatusAndSetting()
        {
            var service = new AccountService(_context);
            var settingId = Guid.NewGuid();

            var setting = new AccountSetting { AccountSettingId = settingId, SettingName = "Setting 1", MaxRiskPerTrade = 10, UntradablePeriodMinutes = 10 };
            await _context.AccountSettings.AddAsync(setting);

            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId,
                    AccountName = "Account 1",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT5",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId,
                    AccountName = "Account 2",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Failed,
                    AccountSettingsId = settingId,
                    AccountName = "Account 3",
                    BrokerLogin = "login3",
                    BrokerPassword = "pass3",
                    BrokerServer = "server3",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 3000,
                    HighestCapital = 3000,
                    InitialCapital = 3000,
                    LowestCapital = 3000,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.FilterAccountsAsync(null, AccountStatus.Active, settingId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.All(a => a.Status == AccountStatus.Active), Is.True);
            Assert.That(result.All(a => a.AccountSettingsId == settingId), Is.True);
        }

        [Test]
        public async Task FilterAccountsAsync_WithNoStatusFilter_FiltersOnlyByPlatformAndSetting()
        {
            var service = new AccountService(_context);
            var settingId = Guid.NewGuid();

            var setting = new AccountSetting { AccountSettingId = settingId, SettingName = "Setting 1", MaxRiskPerTrade = 10, UntradablePeriodMinutes = 10 };
            await _context.AccountSettings.AddAsync(setting);

            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId,
                    AccountName = "Account 1",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Failed,
                    AccountSettingsId = settingId,
                    AccountName = "Account 2",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT5",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId,
                    AccountName = "Account 3",
                    BrokerLogin = "login3",
                    BrokerPassword = "pass3",
                    BrokerServer = "server3",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 3000,
                    HighestCapital = 3000,
                    InitialCapital = 3000,
                    LowestCapital = 3000,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.FilterAccountsAsync("MT4", null, settingId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.All(a => a.Platform == "MT4"), Is.True);
            Assert.That(result.All(a => a.AccountSettingsId == settingId), Is.True);
        }

        [Test]
        public async Task FilterAccountsAsync_WithNoSettingFilter_FiltersOnlyByPlatformAndStatus()
        {
            var service = new AccountService(_context);
            var settingId1 = Guid.NewGuid();
            var settingId2 = Guid.NewGuid();

            var settings = new List<AccountSetting>
            {
                new AccountSetting { AccountSettingId = settingId1, SettingName = "Setting 1", MaxRiskPerTrade = 10, UntradablePeriodMinutes = 10 },
                new AccountSetting { AccountSettingId = settingId2, SettingName = "Setting 2", MaxRiskPerTrade = 20, UntradablePeriodMinutes = 20 }
            };
            await _context.AccountSettings.AddRangeAsync(settings);

            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId1,
                    AccountName = "Account 1",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId2,
                    AccountName = "Account 2",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Failed,
                    AccountSettingsId = settingId1,
                    AccountName = "Account 3",
                    BrokerLogin = "login3",
                    BrokerPassword = "pass3",
                    BrokerServer = "server3",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 3000,
                    HighestCapital = 3000,
                    InitialCapital = 3000,
                    LowestCapital = 3000,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.FilterAccountsAsync("MT4", AccountStatus.Active, null);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.All(a => a.Platform == "MT4"), Is.True);
            Assert.That(result.All(a => a.Status == AccountStatus.Active), Is.True);
        }

        [Test]
        public async Task FilterAccountsAsync_WithNoFilters_ReturnsAllAccounts()
        {
            var service = new AccountService(_context);
            var settingId = Guid.NewGuid();

            var setting = new AccountSetting { AccountSettingId = settingId, SettingName = "Setting 1", MaxRiskPerTrade = 10, UntradablePeriodMinutes = 10 };
            await _context.AccountSettings.AddAsync(setting);

            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT4",
                    Status = AccountStatus.Active,
                    AccountSettingsId = settingId,
                    AccountName = "Account 1",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Platform = "MT5",
                    Status = AccountStatus.Failed,
                    AccountSettingsId = settingId,
                    AccountName = "Account 2",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.FilterAccountsAsync(null, null, null);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
        }

        [Test]
        public async Task AllAccountsAsync_ReturnsActiveAccountsWithIps()
        {
            var service = new AccountService(_context);
            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    Status = AccountStatus.Active,
                    Affiliated_IP = "192.168.1.1",
                    AccountName = "Account 1",
                    Platform = "MT4",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Status = AccountStatus.Active,
                    Affiliated_IP = "192.168.1.2",
                    AccountName = "Account 2",
                    Platform = "MT5",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    UserId = Guid.NewGuid()
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    Status = AccountStatus.Failed,
                    Affiliated_IP = "192.168.1.3",
                    AccountName = "Account 3",
                    Platform = "MT4",
                    BrokerLogin = "login3",
                    BrokerPassword = "pass3",
                    BrokerServer = "server3",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 3000,
                    HighestCapital = 3000,
                    InitialCapital = 3000,
                    LowestCapital = 3000,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.AllAccountsAsync();

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.All(x => accounts.Where(a => a.Status == AccountStatus.Active).Select(a => a.Id).Contains(x.Item1)), Is.True);
        }

        [Test]
        public async Task GetActiveAccountsPerUserIdAsync_CountsActiveAccountsForUser()
        {
            var service = new AccountService(_context);
            var userId = Guid.NewGuid();

            var accounts = new List<Account>
            {
                new Account
                {
                    Id = Guid.NewGuid(),
                    UserId = userId,
                    Status = AccountStatus.Active,
                    AccountName = "Account 1",
                    Platform = "MT4",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    UserId = userId,
                    Status = AccountStatus.Active,
                    AccountName = "Account 2",
                    Platform = "MT5",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    UserId = userId,
                    Status = AccountStatus.Failed,
                    AccountName = "Account 3",
                    Platform = "MT4",
                    BrokerLogin = "login3",
                    BrokerPassword = "pass3",
                    BrokerServer = "server3",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 3000,
                    HighestCapital = 3000,
                    InitialCapital = 3000,
                    LowestCapital = 3000
                },
                new Account
                {
                    Id = Guid.NewGuid(),
                    UserId = Guid.NewGuid(),
                    Status = AccountStatus.Active,
                    AccountName = "Account 4",
                    Platform = "MT5",
                    BrokerLogin = "login4",
                    BrokerPassword = "pass4",
                    BrokerServer = "server4",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 4000,
                    HighestCapital = 4000,
                    InitialCapital = 4000,
                    LowestCapital = 4000
                }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.SaveChangesAsync();

            var result = await service.GetActiveAccountsPerUserIdAsync(userId);

            Assert.That(result, Is.EqualTo(2));
        }

        [Test]
        public async Task GetAccountWithPositionsAsync_IncludesPositionsAndSettings()
        {
            var service = new AccountService(_context);
            var accountId = Guid.NewGuid();
            var settingId = Guid.NewGuid();

            var setting = new AccountSetting { AccountSettingId = settingId, SettingName = "Test Setting", MaxRiskPerTrade = 10, UntradablePeriodMinutes = 10 };
            var account = new Account
            {
                Id = accountId,
                AccountName = "Test Account",
                AccountSettingsId = settingId,
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };
            var positions = new List<Position>
            {
                new Position { Id = "POS1", AccountId = accountId, Symbol = "AAPL", Type = "Buy", OpenedAt = DateTime.UtcNow, Size = 100, Risk = 1 },
                new Position { Id = "POS2", AccountId = accountId, Symbol = "MSFT", Type = "Sell", OpenedAt = DateTime.UtcNow, Size = 200, Risk = 2 }
            };

            await _context.AccountSettings.AddAsync(setting);
            await _context.Accounts.AddAsync(account);
            await _context.Positions.AddRangeAsync(positions);
            await _context.SaveChangesAsync();

            var result = await service.GetAccountWithPositionsAsync(accountId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.AccountPositions, Is.Not.Null);
            Assert.That(result.AccountPositions.Count, Is.EqualTo(2));
            Assert.That(result.AccountSetting, Is.Not.Null);
        }

        [Test]
        public async Task GetAccountWithPositionsAsync_AccountDoesNotExist_ReturnsNull()
        {
            var service = new AccountService(_context);
            var result = await service.GetAccountWithPositionsAsync(Guid.NewGuid());
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task GetAllSettingsAsync_ReturnsAllSettings()
        {
            var service = new AccountSettingService(_context);
            var user1 = new User { Id = Guid.NewGuid(), UserName = "User1" };
            var user2 = new User { Id = Guid.NewGuid(), UserName = "User2" };

            await _context.Users.AddRangeAsync(new[] { user1, user2 });

            var settings = new List<AccountSetting>
            {
                new AccountSetting {
                    AccountSettingId = Guid.NewGuid(),
                    SettingName = "Setting 1",
                    UserId = user1.Id,
                    MaxRiskPerTrade = 10,
                    UntradablePeriodMinutes = 10
                },
                new AccountSetting {
                    AccountSettingId = Guid.NewGuid(),
                    SettingName = "Setting 2",
                    UserId = user2.Id,
                    MaxRiskPerTrade = 20,
                    UntradablePeriodMinutes = 20
                }
            };

            await _context.AccountSettings.AddRangeAsync(settings);
            await _context.SaveChangesAsync();

            var result = await service.GetAllSettingsAsync();

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.Any(s => s.SettingName == "Setting 1"), Is.True);
            Assert.That(result.Any(s => s.SettingName == "Setting 2"), Is.True);
        }

        [Test]
        public async Task GetSettingByIdAsync_ExistingId_ReturnsSetting()
        {
            var service = new AccountSettingService(_context);
            var userId = Guid.NewGuid();
            var user = new User { Id = userId, UserName = "TestUser" };
            await _context.Users.AddAsync(user);

            var settingId = Guid.NewGuid();
            var setting = new AccountSetting
            {
                AccountSettingId = settingId,
                SettingName = "Test Setting",
                UserId = userId,
                MaxRiskPerTrade = 10,
                UntradablePeriodMinutes = 10
            };

            await _context.AccountSettings.AddAsync(setting);
            await _context.SaveChangesAsync();

            var result = await service.GetSettingByIdAsync(settingId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.AccountSettingId, Is.EqualTo(settingId));
            Assert.That(result.UserId, Is.EqualTo(userId));
        }

        [Test]
        public async Task GetSettingByIdAsync_NonExistingId_ReturnsNull()
        {
            var service = new AccountSettingService(_context);
            var result = await service.GetSettingByIdAsync(Guid.NewGuid());
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task GetSettingsForUserAsync_ReturnsUserAndGlobalSettings()
        {
            var service = new AccountSettingService(_context);
            var userId = Guid.NewGuid();
            var otherUserId = Guid.NewGuid();

            var settings = new List<AccountSetting>
            {
                new AccountSetting {
                    AccountSettingId = Guid.NewGuid(),
                    SettingName = "User Setting",
                    UserId = userId,
                    MaxRiskPerTrade = 10,
                    UntradablePeriodMinutes = 10
                },
                new AccountSetting {
                    AccountSettingId = Guid.NewGuid(),
                    SettingName = "Global Setting",
                    UserId = null,
                    MaxRiskPerTrade = 20,
                    UntradablePeriodMinutes = 20
                },
                new AccountSetting {
                    AccountSettingId = Guid.NewGuid(),
                    SettingName = "Other User Setting",
                    UserId = otherUserId,
                    MaxRiskPerTrade = 30,
                    UntradablePeriodMinutes = 30
                }
            };

            await _context.AccountSettings.AddRangeAsync(settings);
            await _context.SaveChangesAsync();

            var result = await service.GetSettingsForUserAsync(userId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.Any(s => s.UserId == userId), Is.True);
            Assert.That(result.Any(s => s.UserId == null), Is.True);
        }

        [Test]
        public async Task CreateSettingAsync_AddsNewSetting()
        {
            var service = new AccountSettingService(_context);
            var settingId = Guid.NewGuid();
            var setting = new AccountSetting
            {
                AccountSettingId = settingId,
                SettingName = "New Setting",
                MaxRiskPerTrade = 10,
                UntradablePeriodMinutes = 10
            };

            await service.CreateSettingAsync(setting);

            var result = await _context.AccountSettings.FirstOrDefaultAsync(s => s.AccountSettingId == settingId);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.SettingName, Is.EqualTo("New Setting"));
        }

        [Test]
        public async Task UpdateSettingAsync_UpdatesExistingSetting()
        {
            var service = new AccountSettingService(_context);
            var settingId = Guid.NewGuid();
            var setting = new AccountSetting
            {
                AccountSettingId = settingId,
                SettingName = "Original Name",
                MaxRiskPerTrade = 10,
                UntradablePeriodMinutes = 10
            };
            await _context.AccountSettings.AddAsync(setting);
            await _context.SaveChangesAsync();

            setting.SettingName = "Updated Name";
            await service.UpdateSettingAsync(setting);

            var result = await _context.AccountSettings.FirstOrDefaultAsync(s => s.AccountSettingId == settingId);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.SettingName, Is.EqualTo("Updated Name"));
        }

        [Test]
        public async Task DeleteSettingAsync_ExistingSetting_DeletesSetting()
        {
            var service = new AccountSettingService(_context);
            var settingId = Guid.NewGuid();
            var setting = new AccountSetting
            {
                AccountSettingId = settingId,
                SettingName = "Test Setting",
                MaxRiskPerTrade = 10,
                UntradablePeriodMinutes = 10
            };
            await _context.AccountSettings.AddAsync(setting);
            await _context.SaveChangesAsync();

            await service.DeleteSettingAsync(settingId);

            var result = await _context.AccountSettings.FirstOrDefaultAsync(s => s.AccountSettingId == settingId);
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task DeleteSettingAsync_NonExistingSetting_DoesNothing()
        {
            var service = new AccountSettingService(_context);
            var initialCount = await _context.AccountSettings.CountAsync();

            await service.DeleteSettingAsync(Guid.NewGuid());

            var finalCount = await _context.AccountSettings.CountAsync();
            Assert.That(finalCount, Is.EqualTo(initialCount));
        }

        [Test]
        public async Task GetAllNotificationsAsync_ReturnsAllNotifications()
        {
            var service = new NotificationService(_context);
            var user = new User { Id = Guid.NewGuid(), UserName = "TestUser" };
            await _context.Users.AddAsync(user);

            var notifications = new List<Notification>
            {
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 1",
                    Description = "Description 1",
                    DateTime = DateTime.UtcNow
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 2",
                    Description = "Description 2",
                    DateTime = DateTime.UtcNow
                }
            };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notifications[0].Id,
                UserId = user.Id
            };

            await _context.Notifications.AddRangeAsync(notifications);
            await _context.NotificationUsers.AddAsync(notificationUser);
            await _context.SaveChangesAsync();

            var result = await service.GetAllNotificationsAsync();

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
            Assert.That(result.First().NotificationUsers.Count, Is.EqualTo(1));
        }

        [Test]
        public async Task GetNotificationByIdAsync_ExistingId_ReturnsNotification()
        {
            var service = new NotificationService(_context);
            var user = new User { Id = Guid.NewGuid(), UserName = "TestUser" };
            await _context.Users.AddAsync(user);

            var notificationId = Guid.NewGuid();
            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                DateTime = DateTime.UtcNow
            };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notificationId,
                UserId = user.Id
            };

            await _context.Notifications.AddAsync(notification);
            await _context.NotificationUsers.AddAsync(notificationUser);
            await _context.SaveChangesAsync();

            var result = await service.GetNotificationByIdAsync(notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Id, Is.EqualTo(notificationId));
            Assert.That(result.NotificationUsers.Count, Is.EqualTo(1));
        }

        [Test]
        public async Task GetNotificationByIdAsync_NonExistingId_ReturnsNull()
        {
            var service = new NotificationService(_context);
            var result = await service.GetNotificationByIdAsync(Guid.NewGuid());
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task CreateNotificationAsync_WithUserIds_CreatesNotificationForUsers()
        {
            var service = new NotificationService(_context);
            var userId1 = Guid.NewGuid();
            var userId2 = Guid.NewGuid();

            await _context.Users.AddRangeAsync(new List<User>
            {
                new User { Id = userId1, UserName = "User1" },
                new User { Id = userId2, UserName = "User2" }
            });
            await _context.SaveChangesAsync();

            var notificationId = Guid.NewGuid();
            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                DateTime = DateTime.UtcNow
            };

            var userIds = new List<Guid> { userId1, userId2 };

            await service.CreateNotificationAsync(notification, userIds);

            var result = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Title, Is.EqualTo("Test Notification"));
            Assert.That(result.Description, Is.EqualTo("Test Description"));
            Assert.That(result.NotificationUsers, Is.Not.Null);
            Assert.That(result.NotificationUsers.Count, Is.EqualTo(2));
            Assert.That(result.NotificationUsers.All(nu => userIds.Contains(nu.UserId)), Is.True);
            Assert.That(result.NotificationUsers.All(nu => nu.IsRead == false), Is.True);
            Assert.That(result.NotificationUsers.All(nu => nu.ReadAt == null), Is.True);
        }

        [Test]
        public async Task CreateNotificationAsync_WithNullUserIds_CreatesNotificationWithNoUsers()
        {
            var service = new NotificationService(_context);
            var notificationId = Guid.NewGuid();
            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                DateTime = DateTime.UtcNow
            };

            await service.CreateNotificationAsync(notification, null);

            var result = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.NotificationUsers, Is.Not.Null);
            Assert.That(result.NotificationUsers, Is.Empty);
        }

        [Test]
        public async Task CreateNotificationAsync_WithEmptyUserIds_CreatesNotificationWithNoUsers()
        {
            var service = new NotificationService(_context);
            var notificationId = Guid.NewGuid();
            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                DateTime = DateTime.UtcNow
            };

            await service.CreateNotificationAsync(notification, new List<Guid>());

            var result = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.NotificationUsers, Is.Not.Null);
            Assert.That(result.NotificationUsers, Is.Empty);
        }

        [Test]
        public async Task CreateNotificationAsync_WithNullNotification_ThrowsArgumentNullException()
        {
            var service = new NotificationService(_context);
            Assert.ThrowsAsync<ArgumentNullException>(() => service.CreateNotificationAsync(null, new List<Guid>()));
        }

        [Test]
        public async Task CreateNotificationEveryoneAsync_CreatesNotificationForAllUsers()
        {
            var service = new NotificationService(_context);
            var users = new List<User>
            {
                new User { Id = Guid.NewGuid(), UserName = "User1" },
                new User { Id = Guid.NewGuid(), UserName = "User2" },
                new User { Id = Guid.NewGuid(), UserName = "User3" }
            };

            await _context.Users.AddRangeAsync(users);
            await _context.SaveChangesAsync();

            var notificationId = Guid.NewGuid();
            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                DateTime = DateTime.UtcNow
            };

            await service.CreateNotificationEveryoneAsync(notification);

            var result = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.NotificationUsers, Is.Not.Null);
            Assert.That(result.NotificationUsers.Count, Is.EqualTo(3));
            Assert.That(result.NotificationUsers.Select(nu => nu.UserId).ToList(),
                Is.EquivalentTo(users.Select(u => u.Id)));
        }

        [Test]
        public async Task CreateNotificationEveryoneAsync_WithNullNotification_ThrowsArgumentNullException()
        {
            var service = new NotificationService(_context);
            Assert.ThrowsAsync<ArgumentNullException>(() => service.CreateNotificationEveryoneAsync(null));
        }

        [Test]
        public async Task UpdateNotificationAsync_ExistingNotification_UpdatesNotification()
        {
            var service = new NotificationService(_context);
            var notificationId = Guid.NewGuid();
            var userId1 = Guid.NewGuid();
            var userId2 = Guid.NewGuid();

            await _context.Users.AddRangeAsync(new List<User>
            {
                new User { Id = userId1, UserName = "User1" },
                new User { Id = userId2, UserName = "User2" }
            });

            var notification = new Notification
            {
                Id = notificationId,
                Title = "Original Title",
                Description = "Original Description",
                DateTime = DateTime.UtcNow.AddDays(-1),
                NotificationUsers = new List<NotificationUser>()
            };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notificationId,
                UserId = userId1
            };

            notification.NotificationUsers.Add(notificationUser);

            await _context.Notifications.AddAsync(notification);
            await _context.SaveChangesAsync();

            var updatedNotification = new Notification
            {
                Id = notificationId,
                Title = "Updated Title",
                Description = "Updated Description",
                DateTime = DateTime.UtcNow
            };

            await service.UpdateNotificationAsync(updatedNotification, new List<Guid> { userId2 });

            var result = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Title, Is.EqualTo("Updated Title"));
            Assert.That(result.Description, Is.EqualTo("Updated Description"));
            Assert.That(result.NotificationUsers, Is.Not.Null);
            Assert.That(result.NotificationUsers.Count, Is.EqualTo(1));
            Assert.That(result.NotificationUsers.First().UserId, Is.EqualTo(userId2));
        }

        [Test]
        public async Task UpdateNotificationAsync_NonExistingNotification_DoesNothing()
        {
            var service = new NotificationService(_context);
            var notificationId = Guid.NewGuid();
            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                DateTime = DateTime.UtcNow
            };

            await service.UpdateNotificationAsync(notification, new List<Guid> { Guid.NewGuid() });

            var result = await _context.Notifications.FirstOrDefaultAsync(n => n.Id == notificationId);
            Assert.That(result, Is.Null);
        }

        [Test]
        public async Task UpdateNotificationAsync_WithNullNotification_ThrowsArgumentNullException()
        {
            var service = new NotificationService(_context);
            Assert.ThrowsAsync<ArgumentNullException>(() => service.UpdateNotificationAsync(null, new List<Guid>()));
        }

        [Test]
        public async Task UpdateNotificationAsync_WithNullUserIds_ClearsUsers()
        {
            var service = new NotificationService(_context);
            var notificationId = Guid.NewGuid();
            var userId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notification = new Notification
            {
                Id = notificationId,
                Title = "Original Title",
                Description = "Original Description",
                DateTime = DateTime.UtcNow,
                NotificationUsers = new List<NotificationUser>()
            };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notificationId,
                UserId = userId
            };

            notification.NotificationUsers.Add(notificationUser);

            await _context.Notifications.AddAsync(notification);
            await _context.SaveChangesAsync();

            var updatedNotification = new Notification
            {
                Id = notificationId,
                Title = "Updated Title",
                Description = "Updated Description",
                DateTime = DateTime.UtcNow
            };

            await service.UpdateNotificationAsync(updatedNotification, null);

            var result = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Title, Is.EqualTo("Updated Title"));
            Assert.That(result.NotificationUsers, Is.Not.Null);
            Assert.That(result.NotificationUsers, Is.Empty);
        }

        [Test]
        public async Task DeleteNotificationAsync_ExistingNotification_DeletesNotification()
        {
            var service = new NotificationService(_context);
            var notificationId = Guid.NewGuid();
            var userId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                DateTime = DateTime.UtcNow,
                NotificationUsers = new List<NotificationUser>()
            };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notificationId,
                UserId = userId
            };

            notification.NotificationUsers.Add(notificationUser);

            await _context.Notifications.AddAsync(notification);
            await _context.SaveChangesAsync();

            await service.DeleteNotificationAsync(notificationId);

            var result = await _context.Notifications.FirstOrDefaultAsync(n => n.Id == notificationId);
            var notificationUsers = await _context.NotificationUsers.Where(nu => nu.NotificationId == notificationId).ToListAsync();

            Assert.That(result, Is.Null);
            Assert.That(notificationUsers, Is.Empty);
        }

        [Test]
        public async Task DeleteNotificationAsync_NonExistingNotification_DoesNothing()
        {
            var service = new NotificationService(_context);
            var initialCount = await _context.Notifications.CountAsync();

            await service.DeleteNotificationAsync(Guid.NewGuid());

            var finalCount = await _context.Notifications.CountAsync();
            Assert.That(finalCount, Is.EqualTo(initialCount));
        }

        [Test]
        public async Task GetUserNotificationsUserAsync_ReturnsNotificationUsersForUser()
        {
            var service = new NotificationService(_context);
            var userId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notifications = new List<Notification>
            {
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 1",
                    Description = "Description 1",
                    DateTime = DateTime.UtcNow.AddDays(-1)
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 2",
                    Description = "Description 2",
                    DateTime = DateTime.UtcNow
                }
            };

            await _context.Notifications.AddRangeAsync(notifications);

            var notificationUsers = new List<NotificationUser>
            {
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[0].Id,
                    UserId = userId
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[1].Id,
                    UserId = userId
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[0].Id,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.NotificationUsers.AddRangeAsync(notificationUsers);
            await _context.SaveChangesAsync();

            var result = await service.GetUserNotificationsUserAsync(userId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
        }

        [Test]
        public async Task GetUserNotificationsUserAsync_WithNullUserId_ReturnsEmptyList()
        {
            var service = new NotificationService(_context);
            var notifications = new List<Notification>
            {
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 1",
                    Description = "Description 1",
                    DateTime = DateTime.UtcNow
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 2",
                    Description = "Description 2",
                    DateTime = DateTime.UtcNow
                }
            };

            await _context.Notifications.AddRangeAsync(notifications);
            await _context.SaveChangesAsync();

            var result = await service.GetUserNotificationsUserAsync(null);

            Assert.That(result, Is.Not.Null);
            Assert.That(result, Is.Empty);
        }

        [Test]
        public async Task GetUserNotificationsAsync_ReturnsNotificationsForUser()
        {
            var service = new NotificationService(_context);
            var userId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notifications = new List<Notification>
            {
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 1",
                    Description = "Description 1",
                    DateTime = DateTime.UtcNow.AddDays(-1)
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 2",
                    Description = "Description 2",
                    DateTime = DateTime.UtcNow
                }
            };

            await _context.Notifications.AddRangeAsync(notifications);

            var notificationUsers = new List<NotificationUser>
            {
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[0].Id,
                    UserId = userId
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[1].Id,
                    UserId = userId
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[0].Id,
                    UserId = Guid.NewGuid()
                }
            };

            await _context.NotificationUsers.AddRangeAsync(notificationUsers);
            await _context.SaveChangesAsync();

            var result = await service.GetUserNotificationsAsync(userId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(2));
        }

        [Test]
        public async Task GetUserUnreadNotificationsAsync_ReturnsUnreadNotificationsForUser()
        {
            var service = new NotificationService(_context);
            var userId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notifications = new List<Notification>
            {
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 1",
                    Description = "Description 1",
                    DateTime = DateTime.UtcNow.AddDays(-1)
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 2",
                    Description = "Description 2",
                    DateTime = DateTime.UtcNow
                }
            };

            await _context.Notifications.AddRangeAsync(notifications);

            var notificationUsers = new List<NotificationUser>
            {
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[0].Id,
                    UserId = userId,
                    IsRead = false
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[1].Id,
                    UserId = userId,
                    IsRead = true
                }
            };

            await _context.NotificationUsers.AddRangeAsync(notificationUsers);
            await _context.SaveChangesAsync();

            var result = await service.GetUserUnreadNotificationsAsync(userId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count(), Is.EqualTo(1));
            Assert.That(result.First().Title, Is.EqualTo("Notification 1"));
        }

        [Test]
        public async Task MarkNotificationAsReadAsync_ForUnreadNotification_MarksAsRead()
        {
            var service = new NotificationService(_context);
            var userId = Guid.NewGuid();
            var notificationId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                TotalReadCount = 0,
                DateTime = DateTime.UtcNow
            };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notificationId,
                UserId = userId,
                IsRead = false,
                ReadAt = null
            };

            await _context.Notifications.AddAsync(notification);
            await _context.NotificationUsers.AddAsync(notificationUser);
            await _context.SaveChangesAsync();

            await service.MarkNotificationAsReadAsync(userId, notificationId);

            var result = await _context.NotificationUsers.FirstOrDefaultAsync(nu => nu.NotificationId == notificationId && nu.UserId == userId);
            var updatedNotification = await _context.Notifications.FirstOrDefaultAsync(n => n.Id == notificationId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.IsRead, Is.True);
            Assert.That(result.ReadAt, Is.Not.Null);
            Assert.That(updatedNotification.TotalReadCount, Is.EqualTo(1));
        }

        [Test]
        public async Task MarkNotificationAsReadAsync_ForAlreadyReadNotification_DoesNotUpdateReadCount()
        {
            var service = new NotificationService(_context);
            var userId = Guid.NewGuid();
            var notificationId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notification = new Notification
            {
                Id = notificationId,
                Title = "Test Notification",
                Description = "Test Description",
                TotalReadCount = 1,
                DateTime = DateTime.UtcNow
            };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notificationId,
                UserId = userId,
                IsRead = true,
                ReadAt = DateTime.UtcNow.AddDays(-1)
            };

            await _context.Notifications.AddAsync(notification);
            await _context.NotificationUsers.AddAsync(notificationUser);
            await _context.SaveChangesAsync();

            await service.MarkNotificationAsReadAsync(userId, notificationId);

            var updatedNotification = await _context.Notifications.FirstOrDefaultAsync(n => n.Id == notificationId);
            Assert.That(updatedNotification.TotalReadCount, Is.EqualTo(1));
        }

        [Test]
        public async Task MarkNotificationAsReadAsync_NonExistingNotification_DoesNothing()
        {
            var service = new NotificationService(_context);
            var initialReadCounts = await _context.Notifications.Select(n => n.TotalReadCount).ToListAsync();

            await service.MarkNotificationAsReadAsync(Guid.NewGuid(), Guid.NewGuid());

            var finalReadCounts = await _context.Notifications.Select(n => n.TotalReadCount).ToListAsync();
            Assert.That(finalReadCounts, Is.EquivalentTo(initialReadCounts));
        }

        [Test]
        public async Task MarkAllNotificationsAsReadAsync_MarksAllUnreadNotificationsAsRead()
        {
            var service = new NotificationService(_context);
            var userId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notifications = new List<Notification>
            {
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 1",
                    Description = "Description 1",
                    TotalReadCount = 0,
                    DateTime = DateTime.UtcNow
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 2",
                    Description = "Description 2",
                    TotalReadCount = 0,
                    DateTime = DateTime.UtcNow
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 3",
                    Description = "Description 3",
                    TotalReadCount = 1,
                    DateTime = DateTime.UtcNow
                }
            };

            await _context.Notifications.AddRangeAsync(notifications);

            var notificationUsers = new List<NotificationUser>
            {
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[0].Id,
                    UserId = userId,
                    IsRead = false
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[1].Id,
                    UserId = userId,
                    IsRead = false
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[2].Id,
                    UserId = userId,
                    IsRead = true,
                    ReadAt = DateTime.UtcNow.AddDays(-1)
                }
            };

            await _context.NotificationUsers.AddRangeAsync(notificationUsers);
            await _context.SaveChangesAsync();

            await service.MarkAllNotificationsAsReadAsync(userId);

            var results = await _context.NotificationUsers.Where(nu => nu.UserId == userId).ToListAsync();
            var updatedNotifications = await _context.Notifications.ToListAsync();

            Assert.That(results, Is.Not.Null);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results.All(nu => nu.IsRead), Is.True);
            Assert.That(results.All(nu => nu.ReadAt.HasValue), Is.True);

            var totalReadAfter = updatedNotifications.Sum(n => n.TotalReadCount);
            Assert.That(totalReadAfter, Is.EqualTo(3));
        }

        [Test]
        public async Task MarkAllNotificationsAsReadAsync_NoUnreadNotifications_DoesNotModifyTotalReadCount()
        {
            var service = new NotificationService(_context);
            var userId = Guid.NewGuid();

            await _context.Users.AddAsync(new User { Id = userId, UserName = "User1" });

            var notifications = new List<Notification>
            {
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 1",
                    Description = "Description 1",
                    TotalReadCount = 1,
                    DateTime = DateTime.UtcNow
                },
                new Notification {
                    Id = Guid.NewGuid(),
                    Title = "Notification 2",
                    Description = "Description 2",
                    TotalReadCount = 1,
                    DateTime = DateTime.UtcNow
                }
            };

            await _context.Notifications.AddRangeAsync(notifications);

            var notificationUsers = new List<NotificationUser>
            {
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[0].Id,
                    UserId = userId,
                    IsRead = true,
                    ReadAt = DateTime.UtcNow.AddDays(-1)
                },
                new NotificationUser {
                    Id = Guid.NewGuid(),
                    NotificationId = notifications[1].Id,
                    UserId = userId,
                    IsRead = true,
                    ReadAt = DateTime.UtcNow.AddDays(-1)
                }
            };

            await _context.NotificationUsers.AddRangeAsync(notificationUsers);
            await _context.SaveChangesAsync();

            var initialTotalRead = notifications.Sum(n => n.TotalReadCount);

            await service.MarkAllNotificationsAsReadAsync(userId);

            var updatedNotifications = await _context.Notifications.ToListAsync();
            var finalTotalRead = updatedNotifications.Sum(n => n.TotalReadCount);

            Assert.That(finalTotalRead, Is.EqualTo(initialTotalRead));
        }

        [Test]
        public async Task CreatePositionAsync_CreatesNewPosition()
        {
            var service = new PositionService(_context, _mockLogger.Object);
            var accountId = Guid.NewGuid();
            var account = new Account
            {
                Id = accountId,
                AccountName = "Test Account",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };
            var positionId = "POS123";

            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();

            var openedAt = DateTime.UtcNow;
            await service.CreatePositionAsync("AAPL", "Buy", account, 100, 2, openedAt, positionId);

            var result = await _context.Positions.FirstOrDefaultAsync(p => p.Id == positionId);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Symbol, Is.EqualTo("AAPL"));
            Assert.That(result.Type, Is.EqualTo("Buy"));
            Assert.That(result.AccountId, Is.EqualTo(accountId));
            Assert.That(result.Size, Is.EqualTo(100));
            Assert.That(result.Risk, Is.EqualTo(2));
            Assert.That(result.Result, Is.Null);
            Assert.That(result.OpenedAt, Is.EqualTo(openedAt));
        }

        [Test]
        public async Task ClosePositionAsync_WithMatchingPosition_ClosesPosition()
        {
            var service = new PositionService(_context, _mockLogger.Object);
            var accountId = Guid.NewGuid();
            var ip = "192.168.1.1";
            var account = new Account
            {
                Id = accountId,
                AccountName = "Test Account",
                Affiliated_IP = ip,
                CurrentCapital = 10000,
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                HighestCapital = 10000,
                InitialCapital = 10000,
                LowestCapital = 10000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };

            var position = new Position
            {
                Id = "POS123",
                AccountId = accountId,
                Symbol = "AAPL",
                Type = "Buy",
                Size = 100,
                Risk = 2,
                OpenedAt = DateTime.UtcNow.AddDays(-1)
            };

            await _context.Accounts.AddAsync(account);
            await _context.Positions.AddAsync(position);
            await _context.SaveChangesAsync();

            var closedAt = DateTime.UtcNow;
            await service.ClosePositionAsync("POS123", 500, 10500, closedAt, ip);

            var result = await _context.Positions.FirstOrDefaultAsync(p => p.Id == "POS123");
            var updatedAccount = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Result, Is.EqualTo(500));
            Assert.That(result.ClosedAt, Is.EqualTo(closedAt));
            Assert.That(updatedAccount.CurrentCapital, Is.EqualTo(10500));
        }

        [Test]
        public async Task ClosePositionAsync_WithNonMatchingIp_LogsErrorAndDoesNothing()
        {
            var service = new PositionService(_context, _mockLogger.Object);
            var accountId = Guid.NewGuid();
            var account = new Account
            {
                Id = accountId,
                AccountName = "Test Account",
                Affiliated_IP = "192.168.1.1",
                CurrentCapital = 10000,
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                HighestCapital = 10000,
                InitialCapital = 10000,
                LowestCapital = 10000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };

            var position = new Position
            {
                Id = "POS123",
                AccountId = accountId,
                Symbol = "AAPL",
                Type = "Buy",
                Size = 100,
                Risk = 2,
                OpenedAt = DateTime.UtcNow.AddDays(-1)
            };

            await _context.Accounts.AddAsync(account);
            await _context.Positions.AddAsync(position);
            await _context.SaveChangesAsync();

            await service.ClosePositionAsync("POS123", 500, 10500, DateTime.UtcNow, "192.168.1.2");

            var result = await _context.Positions.FirstOrDefaultAsync(p => p.Id == "POS123");
            var updatedAccount = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);

            Assert.That(result.Result, Is.Null);
            Assert.That(result.ClosedAt, Is.Null);
            Assert.That(updatedAccount.CurrentCapital, Is.EqualTo(10000));

            _mockLogger.Verify(
                x => x.Log(
                    It.IsAny<LogLevel>(),
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("No matching position found")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }

        [Test]
        public async Task ClosePositionAsync_WithMatchingPositionButNoAccount_LogsErrorForAccount()
        {
            var service = new PositionService(_context, _mockLogger.Object);
            var accountId = Guid.NewGuid();
            var ip = "192.168.1.1";

            // Create a complete account with all required properties
            var account = new Account
            {
                Id = accountId,
                Affiliated_IP = ip,
                AccountName = "Test Account",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active,
                UserId = Guid.NewGuid()
            };

            var position = new Position
            {
                Id = "POS123",
                AccountId = accountId,
                Symbol = "AAPL",
                Type = "Buy",
                Size = 100,
                Risk = 2,
                OpenedAt = DateTime.UtcNow.AddDays(-1),
                Account = account
            };

            await _context.Positions.AddAsync(position);
            await _context.SaveChangesAsync();

            // Remove the account from the context to simulate account not found
            _context.Accounts.Remove(account);
            await _context.SaveChangesAsync();

            var closedAt = DateTime.UtcNow;
            await service.ClosePositionAsync("POS123", 500, 10500, closedAt, ip);

            var result = await _context.Positions.FirstOrDefaultAsync(p => p.Id == "POS123");

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Result, Is.EqualTo(500));
            Assert.That(result.ClosedAt, Is.EqualTo(closedAt));

            _mockLogger.Verify(
                x => x.Log(
                    It.IsAny<LogLevel>(),
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("No matching account found")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }

        [Test]
        public async Task GetPositionsOfUserAsync_ReturnsPositionsForUser()
        {
            var service = new PositionService(_context, _mockLogger.Object);
            var userId = Guid.NewGuid();
            var accountId1 = Guid.NewGuid();
            var accountId2 = Guid.NewGuid();

            var accounts = new List<Account>
            {
                new Account {
                    Id = accountId1,
                    UserId = userId,
                    AccountName = "Account 1",
                    Platform = "MT4",
                    BrokerLogin = "login1",
                    BrokerPassword = "pass1",
                    BrokerServer = "server1",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 1000,
                    HighestCapital = 1000,
                    InitialCapital = 1000,
                    LowestCapital = 1000,
                    Status = AccountStatus.Active
                },
                new Account {
                    Id = accountId2,
                    UserId = userId,
                    AccountName = "Account 2",
                    Platform = "MT5",
                    BrokerLogin = "login2",
                    BrokerPassword = "pass2",
                    BrokerServer = "server2",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 2000,
                    HighestCapital = 2000,
                    InitialCapital = 2000,
                    LowestCapital = 2000,
                    Status = AccountStatus.Active
                },
                new Account {
                    Id = Guid.NewGuid(),
                    UserId = Guid.NewGuid(),
                    AccountName = "Account 3",
                    Platform = "MT4",
                    BrokerLogin = "login3",
                    BrokerPassword = "pass3",
                    BrokerServer = "server3",
                    CreatedAt = DateTime.UtcNow,
                    CurrentCapital = 3000,
                    HighestCapital = 3000,
                    InitialCapital = 3000,
                    LowestCapital = 3000,
                    Status = AccountStatus.Active
                }
            };

            var positions = new List<Position>
            {
                new Position { Id = "POS1", AccountId = accountId1, Symbol = "AAPL", Type = "Buy", OpenedAt = DateTime.UtcNow, Size = 100, Risk = 1 },
                new Position { Id = "POS2", AccountId = accountId2, Symbol = "MSFT", Type = "Buy", OpenedAt = DateTime.UtcNow, Size = 200, Risk = 2 },
                new Position { Id = "POS3", AccountId = accounts[2].Id, Symbol = "GOOG", Type = "Buy", OpenedAt = DateTime.UtcNow, Size = 300, Risk = 3 }
            };

            await _context.Accounts.AddRangeAsync(accounts);
            await _context.Positions.AddRangeAsync(positions);
            await _context.SaveChangesAsync();

            var result = await service.GetPositionsOfUserAsync(userId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Count, Is.EqualTo(2));
            Assert.That(result.Any(p => p.Id == "POS1"), Is.True);
            Assert.That(result.Any(p => p.Id == "POS2"), Is.True);
        }

        [Test]
        public async Task GetPositionsOfUserAsync_UserWithNoPositions_ReturnsEmptyList()
        {
            var service = new PositionService(_context, _mockLogger.Object);
            var userId = Guid.NewGuid();
            var otherUserId = Guid.NewGuid();

            var account = new Account
            {
                Id = Guid.NewGuid(),
                UserId = otherUserId,
                AccountName = "Account 1",
                Platform = "MT4",
                BrokerLogin = "login1",
                BrokerPassword = "pass1",
                BrokerServer = "server1",
                CreatedAt = DateTime.UtcNow,
                CurrentCapital = 1000,
                HighestCapital = 1000,
                InitialCapital = 1000,
                LowestCapital = 1000,
                Status = AccountStatus.Active
            };
            var position = new Position { Id = "POS1", AccountId = account.Id, Symbol = "AAPL", Type = "Buy", OpenedAt = DateTime.UtcNow, Size = 100, Risk = 1 };

            await _context.Accounts.AddAsync(account);
            await _context.Positions.AddAsync(position);
            await _context.SaveChangesAsync();

            var result = await service.GetPositionsOfUserAsync(userId);

            Assert.That(result, Is.Not.Null);
            Assert.That(result, Is.Empty);
        }
    }
}