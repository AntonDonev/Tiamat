using NUnit.Framework;
using Tiamat.Models;
using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.EntityFrameworkCore;
using Tiamat.DataAccess;
using Microsoft.AspNetCore.Identity;

namespace Tiamat.Tests
{
    [TestFixture]
    public class EntityTests
    {

        [Test]
        public void Notification_Constructor_InitializesNotificationUsersCollection()
        {
            var notification = new Notification();

            Assert.That(notification.NotificationUsers, Is.Not.Null);
            Assert.That(notification.NotificationUsers, Is.InstanceOf<List<NotificationUser>>());
            Assert.That(notification.NotificationUsers, Is.Empty);
        }

        [Test]
        public void Notification_Properties_CanBeSetAndRetrieved()
        {
            var id = Guid.NewGuid();
            var notification = new Notification
            {
                Id = id,
                Title = "Test Title",
                Description = "Test Description",
                TotalReadCount = 10,
                DateTime = new DateTime(2023, 1, 1)
            };

            Assert.That(notification.Id, Is.EqualTo(id));
            Assert.That(notification.Title, Is.EqualTo("Test Title"));
            Assert.That(notification.Description, Is.EqualTo("Test Description"));
            Assert.That(notification.TotalReadCount, Is.EqualTo(10));
            Assert.That(notification.DateTime, Is.EqualTo(new DateTime(2023, 1, 1)));
        }

        [Test]
        public void NotificationUser_Properties_CanBeSetAndRetrieved()
        {
            var id = Guid.NewGuid();
            var notification = new Notification { Id = Guid.NewGuid() };
            var user = new User { Id = Guid.NewGuid() };
            var readAt = new DateTime(2023, 1, 1);

            var notificationUser = new NotificationUser
            {
                Id = id,
                NotificationId = notification.Id,
                Notification = notification,
                UserId = user.Id,
                User = user,
                IsRead = true,
                ReadAt = readAt
            };

            Assert.That(notificationUser.Id, Is.EqualTo(id));
            Assert.That(notificationUser.NotificationId, Is.EqualTo(notification.Id));
            Assert.That(notificationUser.Notification, Is.SameAs(notification));
            Assert.That(notificationUser.UserId, Is.EqualTo(user.Id));
            Assert.That(notificationUser.User, Is.SameAs(user));
            Assert.That(notificationUser.IsRead, Is.True);
            Assert.That(notificationUser.ReadAt, Is.EqualTo(readAt));
        }

        [Test]
        public void Position_DefaultConstructor_InitializesProperties()
        {
            var position = new Position();

            Assert.That(position, Is.Not.Null);
        }

        [Test]
        public void Position_ParameterizedConstructor_InitializesProperties()
        {
            var symbol = "AAPL";
            var account = new Account { Id = Guid.NewGuid() };
            var type = "Buy";
            decimal size = 100;
            decimal risk = 2;
            decimal? result = 50;
            var openedAt = DateTime.UtcNow;
            var closedAt = DateTime.UtcNow.AddDays(1);

            var position = new Position(symbol, account, type, size, risk, result, openedAt, closedAt);

            Assert.That(position.Symbol, Is.EqualTo(symbol));
            Assert.That(position.Account, Is.SameAs(account));
            Assert.That(position.AccountId, Is.EqualTo(account.Id));
            Assert.That(position.Type, Is.EqualTo(type));
            Assert.That(position.Size, Is.EqualTo(size));
            Assert.That(position.Risk, Is.EqualTo(risk));
            Assert.That(position.Result, Is.EqualTo(result));
            Assert.That(position.OpenedAt, Is.EqualTo(openedAt));
            Assert.That(position.ClosedAt, Is.EqualTo(closedAt));
        }

        [Test]
        public void Position_Properties_CanBeSetAndRetrieved()
        {
            var id = "POS123";
            var symbol = "AAPL";
            var type = "Buy";
            var accountId = Guid.NewGuid();
            var account = new Account { Id = accountId };
            decimal size = 100;
            decimal risk = 2;
            decimal? result = 50;
            var openedAt = new DateTime(2023, 1, 1);
            var closedAt = new DateTime(2023, 1, 2);

            var position = new Position
            {
                Id = id,
                Symbol = symbol,
                Type = type,
                AccountId = accountId,
                Account = account,
                Size = size,
                Risk = risk,
                Result = result,
                OpenedAt = openedAt,
                ClosedAt = closedAt
            };

            Assert.That(position.Id, Is.EqualTo(id));
            Assert.That(position.Symbol, Is.EqualTo(symbol));
            Assert.That(position.Type, Is.EqualTo(type));
            Assert.That(position.AccountId, Is.EqualTo(accountId));
            Assert.That(position.Account, Is.SameAs(account));
            Assert.That(position.Size, Is.EqualTo(size));
            Assert.That(position.Risk, Is.EqualTo(risk));
            Assert.That(position.Result, Is.EqualTo(result));
            Assert.That(position.OpenedAt, Is.EqualTo(openedAt));
            Assert.That(position.ClosedAt, Is.EqualTo(closedAt));
        }

        [Test]
        public void User_Constructor_InitializesCollections()
        {
            var user = new User();

            Assert.That(user.AccountSettings, Is.Not.Null);
            Assert.That(user.AccountSettings, Is.InstanceOf<List<AccountSetting>>());
            Assert.That(user.AccountSettings, Is.Empty);

            Assert.That(user.Accounts, Is.Not.Null);
            Assert.That(user.Accounts, Is.InstanceOf<List<Account>>());
            Assert.That(user.Accounts, Is.Empty);

            Assert.That(user.NotificationUsers, Is.Not.Null);
            Assert.That(user.NotificationUsers, Is.InstanceOf<List<NotificationUser>>());
            Assert.That(user.NotificationUsers, Is.Empty);
        }

        [Test]
        public void Account_DefaultConstructor_InitializesPositionsCollection()
        {
            var account = new Account();

            Assert.That(account.AccountPositions, Is.Not.Null);
            Assert.That(account.AccountPositions, Is.InstanceOf<List<Position>>());
            Assert.That(account.AccountPositions, Is.Empty);
        }

        [Test]
        public void Account_ParameterizedConstructor_InitializesProperties()
        {
            var user = new User { Id = Guid.NewGuid() };
            var accountName = "Test Account";
            decimal initialCapital = 10000;
            var accountSetting = new AccountSetting { AccountSettingId = Guid.NewGuid() };
            var platform = "MT4";
            var brokerLogin = "login123";
            var brokerPassword = "password123";
            var brokerServer = "server123";
            var createdAt = DateTime.UtcNow;

            var account = new Account(user, accountName, initialCapital, accountSetting, platform, brokerLogin, brokerPassword, brokerServer, createdAt);

            Assert.That(account.Id, Is.Not.EqualTo(Guid.Empty));
            Assert.That(account.UserId, Is.EqualTo(user.Id));
            Assert.That(account.User, Is.SameAs(user));
            Assert.That(account.AccountName, Is.EqualTo(accountName));
            Assert.That(account.InitialCapital, Is.EqualTo(initialCapital));
            Assert.That(account.CurrentCapital, Is.EqualTo(initialCapital));
            Assert.That(account.HighestCapital, Is.EqualTo(initialCapital));
            Assert.That(account.LowestCapital, Is.EqualTo(initialCapital));
            Assert.That(account.AccountSettingsId, Is.EqualTo(accountSetting.AccountSettingId));
            Assert.That(account.AccountSetting, Is.SameAs(accountSetting));
            Assert.That(account.Status, Is.EqualTo(AccountStatus.Pending));
            Assert.That(account.Platform, Is.EqualTo(platform));
            Assert.That(account.BrokerLogin, Is.EqualTo(brokerLogin));
            Assert.That(account.BrokerPassword, Is.EqualTo(brokerPassword));
            Assert.That(account.BrokerServer, Is.EqualTo(brokerServer));
            Assert.That(account.CreatedAt, Is.EqualTo(createdAt));
            Assert.That(account.AccountPositions, Is.Not.Null);
            Assert.That(account.AccountPositions, Is.Empty);
            Assert.That(account.Affiliated_IP, Is.Null);
        }

        [Test]
        public void Account_Properties_CanBeSetAndRetrieved()
        {
            var id = Guid.NewGuid();
            var userId = Guid.NewGuid();
            var user = new User { Id = userId };
            var accountName = "Test Account";
            var initialCapital = 10000m;
            var currentCapital = 11000m;
            var highestCapital = 12000m;
            var lowestCapital = 9000m;
            var accountSettingsId = Guid.NewGuid();
            var accountSetting = new AccountSetting { AccountSettingId = accountSettingsId };
            var platform = "MT4";
            var brokerLogin = "login123";
            var brokerPassword = "password123";
            var brokerServer = "server123";
            var status = AccountStatus.Active;
            var affiliatedIp = "192.168.1.1";
            var vpsName = "VPS1";
            var adminEmail = "admin@example.com";
            var createdAt = new DateTime(2023, 1, 1);
            var lastUpdatedAt = new DateTime(2023, 1, 2);

            var account = new Account
            {
                Id = id,
                UserId = userId,
                User = user,
                AccountName = accountName,
                InitialCapital = initialCapital,
                CurrentCapital = currentCapital,
                HighestCapital = highestCapital,
                LowestCapital = lowestCapital,
                AccountSettingsId = accountSettingsId,
                AccountSetting = accountSetting,
                Platform = platform,
                BrokerLogin = brokerLogin,
                BrokerPassword = brokerPassword,
                BrokerServer = brokerServer,
                Status = status,
                Affiliated_IP = affiliatedIp,
                VPSName = vpsName,
                AdminEmail = adminEmail,
                CreatedAt = createdAt,
                LastUpdatedAt = lastUpdatedAt
            };

            Assert.That(account.Id, Is.EqualTo(id));
            Assert.That(account.UserId, Is.EqualTo(userId));
            Assert.That(account.User, Is.SameAs(user));
            Assert.That(account.AccountName, Is.EqualTo(accountName));
            Assert.That(account.InitialCapital, Is.EqualTo(initialCapital));
            Assert.That(account.CurrentCapital, Is.EqualTo(currentCapital));
            Assert.That(account.HighestCapital, Is.EqualTo(highestCapital));
            Assert.That(account.LowestCapital, Is.EqualTo(lowestCapital));
            Assert.That(account.AccountSettingsId, Is.EqualTo(accountSettingsId));
            Assert.That(account.AccountSetting, Is.SameAs(accountSetting));
            Assert.That(account.Platform, Is.EqualTo(platform));
            Assert.That(account.BrokerLogin, Is.EqualTo(brokerLogin));
            Assert.That(account.BrokerPassword, Is.EqualTo(brokerPassword));
            Assert.That(account.BrokerServer, Is.EqualTo(brokerServer));
            Assert.That(account.Status, Is.EqualTo(status));
            Assert.That(account.Affiliated_IP, Is.EqualTo(affiliatedIp));
            Assert.That(account.VPSName, Is.EqualTo(vpsName));
            Assert.That(account.AdminEmail, Is.EqualTo(adminEmail));
            Assert.That(account.CreatedAt, Is.EqualTo(createdAt));
            Assert.That(account.LastUpdatedAt, Is.EqualTo(lastUpdatedAt));
        }

        [Test]
        public void AccountSetting_DefaultConstructor_InitializesAccountsCollection()
        {
            var accountSetting = new AccountSetting();

            Assert.That(accountSetting.Accounts, Is.Not.Null);
            Assert.That(accountSetting.Accounts, Is.InstanceOf<List<Account>>());
            Assert.That(accountSetting.Accounts, Is.Empty);
        }

        [Test]
        public void AccountSetting_ParameterizedConstructor_InitializesProperties()
        {
            var settingName = "Conservative";
            int maxRiskPerTrade = 2;
            int untradablePeriodMinutes = 30;
            var user = new User { Id = Guid.NewGuid() };

            var accountSetting = new AccountSetting(settingName, maxRiskPerTrade, untradablePeriodMinutes, user);

            Assert.That(accountSetting.AccountSettingId, Is.Not.EqualTo(Guid.Empty));
            Assert.That(accountSetting.SettingName, Is.EqualTo(settingName));
            Assert.That(accountSetting.MaxRiskPerTrade, Is.EqualTo(maxRiskPerTrade));
            Assert.That(accountSetting.UntradablePeriodMinutes, Is.EqualTo(untradablePeriodMinutes));
            Assert.That(accountSetting.UserId, Is.EqualTo(user.Id));
            Assert.That(accountSetting.User, Is.SameAs(user));
            Assert.That(accountSetting.Accounts, Is.Not.Null);
            Assert.That(accountSetting.Accounts, Is.Empty);
        }

        [Test]
        public void AccountSetting_ParameterizedConstructor_HandlesNullUser()
        {
            var settingName = "Conservative";
            int maxRiskPerTrade = 2;
            int untradablePeriodMinutes = 30;

            var accountSetting = new AccountSetting(settingName, maxRiskPerTrade, untradablePeriodMinutes, null);

            Assert.That(accountSetting.AccountSettingId, Is.Not.EqualTo(Guid.Empty));
            Assert.That(accountSetting.SettingName, Is.EqualTo(settingName));
            Assert.That(accountSetting.MaxRiskPerTrade, Is.EqualTo(maxRiskPerTrade));
            Assert.That(accountSetting.UntradablePeriodMinutes, Is.EqualTo(untradablePeriodMinutes));
            Assert.That(accountSetting.UserId, Is.Null);
            Assert.That(accountSetting.User, Is.Null);
            Assert.That(accountSetting.Accounts, Is.Not.Null);
            Assert.That(accountSetting.Accounts, Is.Empty);
        }

        [Test]
        public void AccountSetting_Properties_CanBeSetAndRetrieved()
        {
            var accountSettingId = Guid.NewGuid();
            var settingName = "Conservative";
            int maxRiskPerTrade = 2;
            int untradablePeriodMinutes = 30;
            var userId = Guid.NewGuid();
            var user = new User { Id = userId };

            var accountSetting = new AccountSetting
            {
                AccountSettingId = accountSettingId,
                SettingName = settingName,
                MaxRiskPerTrade = maxRiskPerTrade,
                UntradablePeriodMinutes = untradablePeriodMinutes,
                UserId = userId,
                User = user
            };

            Assert.That(accountSetting.AccountSettingId, Is.EqualTo(accountSettingId));
            Assert.That(accountSetting.SettingName, Is.EqualTo(settingName));
            Assert.That(accountSetting.MaxRiskPerTrade, Is.EqualTo(maxRiskPerTrade));
            Assert.That(accountSetting.UntradablePeriodMinutes, Is.EqualTo(untradablePeriodMinutes));
            Assert.That(accountSetting.UserId, Is.EqualTo(userId));
            Assert.That(accountSetting.User, Is.SameAs(user));
        }

        [Test]
        public void Application_DefaultValues_AreCorrectlySet()
        {
            var application = new Application();

            Assert.That(application.Status, Is.EqualTo("Pending"));
            Assert.That(application.CreatedAt, Is.EqualTo(DateTime.UtcNow).Within(TimeSpan.FromSeconds(10)));
            Assert.That(application.UpdatedAt, Is.Null);
            Assert.That(application.AdminNotes, Is.Null);
            Assert.That(application.ApprovalDate, Is.Null);
        }

        [Test]
        public void Application_Properties_CanBeSetAndRetrieved()
        {
            var id = Guid.NewGuid();
            var fullName = "John Doe";
            var email = "john.doe@example.com";
            var dob = new DateTime(1980, 1, 1);
            var passport = "AB123456";
            var country = "USA";
            decimal netWorth = 1000000;
            var isPep = false;
            var isAccredited = true;
            var status = "Approved";
            var createdAt = new DateTime(2023, 1, 1);
            var updatedAt = new DateTime(2023, 1, 2);
            var notes = "Good applicant";
            var adminId = Guid.NewGuid();
            var admin = new User { Id = adminId };
            var approvalDate = new DateTime(2023, 1, 3);
            var userId = Guid.NewGuid();
            var user = new User { Id = userId };

            var application = new Application
            {
                Id = id,
                FullName = fullName,
                Email = email,
                DateOfBirth = dob,
                PassportNumber = passport,
                ResidencyCountry = country,
                EstimatedNetWorth = netWorth,
                IsPoliticallyExposedPerson = isPep,
                AccreditedInvestor = isAccredited,
                Status = status,
                CreatedAt = createdAt,
                UpdatedAt = updatedAt,
                AdminNotes = notes,
                AdminId = adminId,
                Admin = admin,
                ApprovalDate = approvalDate,
                IdentityUserId = userId,
                User = user
            };

            Assert.That(application.Id, Is.EqualTo(id));
            Assert.That(application.FullName, Is.EqualTo(fullName));
            Assert.That(application.Email, Is.EqualTo(email));
            Assert.That(application.DateOfBirth, Is.EqualTo(dob));
            Assert.That(application.PassportNumber, Is.EqualTo(passport));
            Assert.That(application.ResidencyCountry, Is.EqualTo(country));
            Assert.That(application.EstimatedNetWorth, Is.EqualTo(netWorth));
            Assert.That(application.IsPoliticallyExposedPerson, Is.EqualTo(isPep));
            Assert.That(application.AccreditedInvestor, Is.EqualTo(isAccredited));
            Assert.That(application.Status, Is.EqualTo(status));
            Assert.That(application.CreatedAt, Is.EqualTo(createdAt));
            Assert.That(application.UpdatedAt, Is.EqualTo(updatedAt));
            Assert.That(application.AdminNotes, Is.EqualTo(notes));
            Assert.That(application.AdminId, Is.EqualTo(adminId));
            Assert.That(application.Admin, Is.SameAs(admin));
            Assert.That(application.ApprovalDate, Is.EqualTo(approvalDate));
            Assert.That(application.IdentityUserId, Is.EqualTo(userId));
            Assert.That(application.User, Is.SameAs(user));
        }

        [Test]
        public void RelationshipTests_NotificationUser_LinksBothEntities()
        {
            var notification = new Notification { Id = Guid.NewGuid() };
            var user = new User { Id = Guid.NewGuid() };

            var notificationUser = new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notification.Id,
                Notification = notification,
                UserId = user.Id,
                User = user,
                IsRead = false
            };

            notification.NotificationUsers.Add(notificationUser);
            user.NotificationUsers.Add(notificationUser);

            Assert.That(notification.NotificationUsers.Count, Is.EqualTo(1));
            Assert.That(notification.NotificationUsers.First(), Is.SameAs(notificationUser));
            Assert.That(user.NotificationUsers.Count, Is.EqualTo(1));
            Assert.That(user.NotificationUsers.First(), Is.SameAs(notificationUser));
        }

        [Test]
        public void RelationshipTests_UserAccount_LinksBothEntities()
        {
            var user = new User { Id = Guid.NewGuid() };
            var account = new Account
            {
                Id = Guid.NewGuid(),
                UserId = user.Id,
                User = user,
                AccountName = "Test Account"
            };

            user.Accounts.Add(account);

            Assert.That(user.Accounts.Count, Is.EqualTo(1));
            Assert.That(user.Accounts.First(), Is.SameAs(account));
            Assert.That(account.User, Is.SameAs(user));
            Assert.That(account.UserId, Is.EqualTo(user.Id));
        }

        [Test]
        public void RelationshipTests_AccountPosition_LinksBothEntities()
        {
            var account = new Account { Id = Guid.NewGuid() };
            var position = new Position
            {
                Id = "POS001",
                AccountId = account.Id,
                Account = account,
                Symbol = "AAPL",
                Type = "Buy",
                Size = 100,
                Risk = 2,
                OpenedAt = DateTime.UtcNow
            };

            account.AccountPositions.Add(position);

            Assert.That(account.AccountPositions.Count, Is.EqualTo(1));
            Assert.That(account.AccountPositions.First(), Is.SameAs(position));
            Assert.That(position.Account, Is.SameAs(account));
            Assert.That(position.AccountId, Is.EqualTo(account.Id));
        }

        [Test]
        public void RelationshipTests_UserAccountSetting_LinksBothEntities()
        {
            var user = new User { Id = Guid.NewGuid() };
            var accountSetting = new AccountSetting
            {
                AccountSettingId = Guid.NewGuid(),
                UserId = user.Id,
                User = user,
                SettingName = "Conservative"
            };

            user.AccountSettings.Add(accountSetting);

            Assert.That(user.AccountSettings.Count, Is.EqualTo(1));
            Assert.That(user.AccountSettings.First(), Is.SameAs(accountSetting));
            Assert.That(accountSetting.User, Is.SameAs(user));
            Assert.That(accountSetting.UserId, Is.EqualTo(user.Id));
        }

        [Test]
        public void RelationshipTests_AccountSettingAccount_LinksBothEntities()
        {
            var accountSetting = new AccountSetting
            {
                AccountSettingId = Guid.NewGuid(),
                SettingName = "Conservative"
            };

            var account = new Account
            {
                Id = Guid.NewGuid(),
                AccountSettingsId = accountSetting.AccountSettingId,
                AccountSetting = accountSetting,
                AccountName = "Test Account"
            };

            accountSetting.Accounts.Add(account);

            Assert.That(accountSetting.Accounts.Count, Is.EqualTo(1));
            Assert.That(accountSetting.Accounts.First(), Is.SameAs(account));
            Assert.That(account.AccountSetting, Is.SameAs(accountSetting));
            Assert.That(account.AccountSettingsId, Is.EqualTo(accountSetting.AccountSettingId));
        }

        [Test]
        public void DbContext_EntityConfiguration_SetsCorrectRelationships()
        {
            var options = new DbContextOptionsBuilder<TiamatDbContext>()
                .UseInMemoryDatabase(databaseName: "TestDatabase")
                .Options;

            using (var context = new TiamatDbContext(options))
            {
                var notificationUserEntity = context.Model.FindEntityType(typeof(NotificationUser));
                Assert.That(notificationUserEntity, Is.Not.Null);

                var accountEntity = context.Model.FindEntityType(typeof(Account));
                var highestCapitalProperty = accountEntity.FindProperty(nameof(Account.HighestCapital));
                Assert.That(highestCapitalProperty, Is.Not.Null);
                Assert.That(highestCapitalProperty.GetPrecision(), Is.EqualTo(18));
                Assert.That(highestCapitalProperty.GetScale(), Is.EqualTo(2));

                Assert.That(context.Accounts, Is.Not.Null);
                Assert.That(context.AccountSettings, Is.Not.Null);
                Assert.That(context.Positions, Is.Not.Null);
                Assert.That(context.Notifications, Is.Not.Null);
                Assert.That(context.NotificationUsers, Is.Not.Null);
            }
        }
    }
}