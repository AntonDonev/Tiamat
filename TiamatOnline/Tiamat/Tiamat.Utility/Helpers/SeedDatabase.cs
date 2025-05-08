using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Tiamat.DataAccess;
using Tiamat.Models;
using Microsoft.EntityFrameworkCore;

namespace Tiamat.Utility.Services
{
    public class SeedDatabase : IHostedService
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly Random _random = new Random();

        public SeedDatabase(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            using var scope = _serviceProvider.CreateScope();

            var roleManager = scope.ServiceProvider.GetRequiredService<RoleManager<IdentityRole<Guid>>>();
            var userManager = scope.ServiceProvider.GetRequiredService<UserManager<User>>();
            var dbContext = scope.ServiceProvider.GetRequiredService<TiamatDbContext>();

            await CreateRoleIfNotExists(roleManager, "admin", new Guid("22222222-2222-2222-2222-222222222222"));
            await CreateRoleIfNotExists(roleManager, "normal", new Guid("33333333-3333-3333-3333-333333333333"));

            var adminUserId = new Guid("44444444-4444-4444-4444-444444444444");
            var normalUserId = new Guid("55555555-5555-5555-5555-555555555555");

            await CreateUserIfNotExists(
                userManager,
                userId: adminUserId,
                userName: "admin",
                email: "admin@tiamat.com",
                password: "Admin123!",
                roleName: "admin"
            );

            await CreateUserIfNotExists(
                userManager,
                userId: normalUserId,
                userName: "user",
                email: "user@tiamat.com",
                password: "User123!",
                roleName: "normal"
            );

            var accountSettingId = new Guid("11111111-1111-1111-1111-111111111111");
            var existingSetting = await dbContext.AccountSettings.FindAsync(accountSettingId);
            if (existingSetting == null)
            {
                dbContext.AccountSettings.Add(new AccountSetting
                {
                    AccountSettingId = accountSettingId,
                    SettingName = "Default (2% / 60m)",
                    MaxRiskPerTrade = 2,
                    UntradablePeriodMinutes = 60,
                    UserId = null
                });
                await dbContext.SaveChangesAsync(cancellationToken);
            }

            // Създаване на тестови търговски данни за нормалния потребител
            await CreateTradingDataForNormalUser(dbContext, normalUserId, cancellationToken, accountSettingId);
        }

        public Task StopAsync(CancellationToken cancellationToken)
            => Task.CompletedTask;

        private static async Task CreateRoleIfNotExists(
            RoleManager<IdentityRole<Guid>> roleManager,
            string roleName,
            Guid fixedRoleId)
        {
            var existingRole = await roleManager.FindByNameAsync(roleName);
            if (existingRole == null)
            {
                var newRole = new IdentityRole<Guid>()
                {
                    Id = fixedRoleId,
                    Name = roleName,
                    NormalizedName = roleName.ToUpper()
                };
                await roleManager.CreateAsync(newRole);
            }
        }

        private static async Task CreateUserIfNotExists(
            UserManager<User> userManager,
            Guid userId,
            string userName,
            string email,
            string password,
            string roleName = null)
        {
            var existingUser = await userManager.FindByNameAsync(userName);
            if (existingUser == null)
            {
                var user = new User
                {
                    Id = userId,
                    UserName = userName,
                    NormalizedUserName = userName.ToUpper(),
                    Email = email,
                    NormalizedEmail = email.ToUpper(),
                    EmailConfirmed = true,
                    SecurityStamp = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    ConcurrencyStamp = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
                };

                await userManager.CreateAsync(user, password);
            }

            if (!string.IsNullOrEmpty(roleName))
            {
                var userInDb = await userManager.FindByNameAsync(userName);
                if (userInDb != null && !(await userManager.IsInRoleAsync(userInDb, roleName)))
                {
                    await userManager.AddToRoleAsync(userInDb, roleName);
                }
            }
        }

        private async Task CreateTradingDataForNormalUser(TiamatDbContext dbContext, Guid userId, CancellationToken cancellationToken, Guid defaultSetting)
        {

            // Създаване на търговски акаунт за потребителя
            var accountId = new Guid("77777777-7777-7777-7777-777777777777");
            var existingAccount = await dbContext.Accounts.FindAsync(accountId);
            if (existingAccount == null)
            {
                dbContext.Accounts.Add(new Account
                {
                    Id = accountId,
                    UserId = userId,
                    AccountName = "NOIT-25",
                    InitialCapital = 10000m,
                    CurrentCapital = 10500m,
                    HighestCapital = 10800m,
                    LowestCapital = 9800m,
                    AccountSettingsId = defaultSetting,
                    Platform = "MT5",
                    BrokerLogin = "demo12345",
                    BrokerPassword = "pass12345",
                    BrokerServer = "Demo.Server",
                    Status = AccountStatus.Active,
                    Affiliated_HWID = "49AD37E3-15D5-436C-BAA3-1A0177A4BF6D", 
                    VPSName = "VPS_Demo",
                    AdminEmail = "admin@tiamat.com",
                    CreatedAt = DateTime.Now.AddDays(-30),
                    LastUpdatedAt = DateTime.Now
                });
                await dbContext.SaveChangesAsync(cancellationToken);
            }

            // Skip creating positions for accounts
            // await CreatePositionsForAccount(dbContext, accountId, cancellationToken);
        }

        private async Task CreatePositionsForAccount(TiamatDbContext dbContext, Guid accountId, CancellationToken cancellationToken)
        {
            // Проверка дали вече има създадени позиции за този акаунт
            if (await dbContext.Positions.AnyAsync(p => p.AccountId == accountId, cancellationToken))
            {
                return;
            }

            // Генериране на 5 произволни дати в последните 30 дни
            var tradingDates = new List<DateTime>();
            var baseDate = DateTime.Now.AddDays(-30);

            for (int i = 0; i < 5; i++)
            {
                tradingDates.Add(baseDate.AddDays(_random.Next(1, 29)));
            }

            // Сортиране на датите хронологично
            tradingDates.Sort();

            // Инструменти за търговия
            var symbols = new List<string> { "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GOLD", "OIL", "BTCUSD" };

            // Типове позиции
            var positionTypes = new List<string> { "Покупка", "Продажба" };

            // Списък с GUID-ове за позициите
            var positionGuids = new List<Guid>
            {
                new Guid("a1111111-1111-1111-1111-111111111111"),
                new Guid("a2222222-2222-2222-2222-222222222222"),
                new Guid("a3333333-3333-3333-3333-333333333333"),
                new Guid("a4444444-4444-4444-4444-444444444444"),
                new Guid("a5555555-5555-5555-5555-555555555555"),
                new Guid("a6666666-6666-6666-6666-666666666666"),
                new Guid("a7777777-7777-7777-7777-777777777777"),
                new Guid("a8888888-8888-8888-8888-888888888888"),
                new Guid("a9999999-9999-9999-9999-999999999999"),
                new Guid("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                new Guid("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                new Guid("cccccccc-cccc-cccc-cccc-cccccccccccc"),
                new Guid("dddddddd-dddd-dddd-dddd-dddddddddddd"),
                new Guid("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
                new Guid("ffffffff-ffff-ffff-ffff-ffffffffffff")
            };

            // Създаване на 2-3 позиции за всяка дата (общо 10-15 позиции)
            var positions = new List<Position>();
            var positionIndex = 0;

            foreach (var date in tradingDates)
            {
                var positionsPerDay = _random.Next(2, 4); // 2 или 3 позиции на ден

                for (int i = 0; i < positionsPerDay && positionIndex < positionGuids.Count; i++)
                {
                    var symbol = symbols[_random.Next(symbols.Count)];
                    var type = positionTypes[_random.Next(positionTypes.Count)];

                    // Генериране на произволен час за отваряне
                    var openHour = _random.Next(9, 16); // Между 9:00 и 15:59
                    var openMinute = _random.Next(0, 60);
                    var openTime = new DateTime(date.Year, date.Month, date.Day, openHour, openMinute, 0);

                    // Генериране на произволен час за затваряне (1-4 часа след отваряне)
                    var closeTime = openTime.AddHours(_random.Next(1, 5)).AddMinutes(_random.Next(0, 60));

                    // Генериране на произволни размери и резултати
                    var size = _random.Next(1, 10) * 0.1m; // Между 0.1 и 0.9
                    var risk = _random.Next(1, 20) * 0.1m; // Между 0.1% и 2.0%

                    // Положителен или отрицателен резултат с по-голям шанс за положителен
                    var isProfit = _random.Next(10) < 7; // 70% шанс за печалба
                    var result = isProfit ? _random.Next(10, 100) : -_random.Next(10, 50);

                    positions.Add(new Position
                    {
                        Id = positionGuids[positionIndex++].ToString(),
                        Symbol = symbol,
                        Type = type,
                        AccountId = accountId,
                        Size = size,
                        Risk = risk,
                        Result = result,
                        OpenedAt = openTime,
                        ClosedAt = closeTime
                    });
                }
            }

            // Добавяне на позициите към базата данни
            dbContext.Positions.AddRange(positions);
            await dbContext.SaveChangesAsync(cancellationToken);
        }
    }
}