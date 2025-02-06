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

namespace Tiamat.Utility.Services
{
    public class SeedDatabase : IHostedService
    {
        private readonly IServiceProvider _serviceProvider;

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
    }
}
