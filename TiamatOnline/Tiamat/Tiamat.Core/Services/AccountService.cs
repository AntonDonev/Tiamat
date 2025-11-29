using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Core.Services.Interfaces;
using Tiamat.DataAccess;
using Tiamat.Models;
using Microsoft.Extensions.Logging;
using System.Diagnostics;

namespace Tiamat.Core.Services
{
    public class AccountService : IAccountService
    {
        private readonly TiamatDbContext _context;
        private readonly INotificationService _notificationService;
        private readonly IPythonApiService _pythonSocketService;
        private readonly ILogger<AccountService> _logger;

        public AccountService(
            TiamatDbContext context,
            INotificationService notificationService,
            IPythonApiService pythonSocketService,
            ILogger<AccountService> logger)
        {
            _context = context;
            _notificationService = notificationService;
            _pythonSocketService = pythonSocketService;
            _logger = logger;
        }
        

        public AccountService(TiamatDbContext context)
        {
            _context = context;

            _notificationService = null;
            _pythonSocketService = null;
            _logger = null;
        }

        public async Task<IEnumerable<Account>> GetAllAccountsAsync()
        {
            return await _context.Accounts
                .Include(a => a.AccountSetting)
                .ToListAsync();
        }

        public async Task<Account> GetAccountByIdAsync(Guid id)
        {
            return await _context.Accounts
                .Include(a => a.AccountSetting)
                .FirstOrDefaultAsync(a => a.Id == id);
        }

        public async Task<Account> GetAccountByHwidAsync(string hwid)
        {
            return await _context.Accounts.FirstOrDefaultAsync(x => x.Affiliated_HWID == hwid);
        }

        public async Task CreateAccountAsync(Account account)
        {
            await _context.Accounts.AddAsync(account);
            await _context.SaveChangesAsync();
        }

        public async Task UpdateAccountAsync(Account account)
        {
            _context.Accounts.Update(account);
            await _context.SaveChangesAsync();
        }

        public async Task DeleteAccountAsync(Guid id)
        {
            var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == id);
            if (account != null)
            {
                _context.Accounts.Remove(account);
                await _context.SaveChangesAsync();
            }
        }

        public async Task<(bool IsSuccess, string ErrorMessage)> DenyAccountAndNotifyAsync(Guid accountId, string title, string message, bool useDefaultMessage)
        {
            try
            {

                if (_notificationService == null)
                    throw new InvalidOperationException("NotificationService is not available. Make sure to use the correct constructor.");

                if (_logger == null)
                    throw new InvalidOperationException("Logger is not available. Make sure to use the correct constructor.");


                await AccountReviewAsync(AccountStatus.Failed, accountId);


                if (useDefaultMessage)
                {
                    message = "След внимателно разглеждане, съжаляваме да ви информираме, че вашето заявление е отхвърлено. Ценим вашия интерес и ако имате въпроси или бихте искали да кандидатствате отново в бъдеще, не се колебайте да се свържете с нашия екип за поддръжка.";
                }


                Notification notification = new Notification
                {
                    Id = Guid.NewGuid(),
                    Title = title,
                    Description = message,
                    DateTime = DateTime.Now
                };

                var account = await GetAccountByIdAsync(accountId);
                List<Guid> target = new List<Guid> { account.UserId };

                await _notificationService.CreateNotificationAsync(notification, target);

                return (true, null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in DenyAccountAndNotifyAsync");
                return (false, ex.Message);
            }
        }

        public async Task AccountReviewAsync(AccountStatus newStatus, Guid accountId, string VPSName, string AdminEmail, string AffiliatedHWID)
        {
            var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            if (account != null)
            {
                account.VPSName = VPSName;
                account.AdminEmail = AdminEmail;
                account.Status = newStatus;
                account.Affiliated_HWID = AffiliatedHWID;

                account.LastUpdatedAt = null;
                await _context.SaveChangesAsync();
            }
        }

        public async Task AccountReviewAsync(AccountStatus newStatus, Guid accountId)
        {
            var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            if (account != null)
            {
                account.Status = newStatus;

                account.LastUpdatedAt = null;
                await _context.SaveChangesAsync();
            }
        }

        public async Task<IEnumerable<Account>> FilterAccountsAsync(string platform, AccountStatus? status, Guid? accountSettingId)
        {
            var query = _context.Accounts.AsQueryable();
            if (!string.IsNullOrEmpty(platform)) query = query.Where(a => a.Platform == platform);
            if (status.HasValue) query = query.Where(a => a.Status == status.Value);
            if (accountSettingId.HasValue) query = query.Where(a => a.AccountSettingsId == accountSettingId.Value);

            return await query
                .Include(a => a.AccountSetting)
                .ToListAsync();
        }

        public async Task<IEnumerable<(Guid, string?)>> AllAccountsAsync()
        {
            var accounts = await _context.Accounts
                .Where(x => x.Status == AccountStatus.Active)
                .Select(x => new { x.Id, x.Affiliated_HWID })
                .ToListAsync();

            return accounts.Select(x => (x.Id, x.Affiliated_HWID)).ToList();
        }

        public async Task<int> GetActiveAccountsPerUserIdAsync(Guid userId)
        {
            return await _context.Accounts
                .Where(x => x.UserId == userId && x.Status == AccountStatus.Active)
                .CountAsync();
        }

        public async Task<Account> GetAccountWithPositionsAsync(Guid id)
        {
            return await _context.Accounts
                .Include(a => a.AccountPositions)
                .Include(a => a.AccountSetting)
                .FirstOrDefaultAsync(a => a.Id == id);
        }

        public async Task ResetHwidAsync(Guid accountId, string newHwid)
        {
            var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            if (account != null)
            {
                account.Affiliated_HWID = newHwid;
                account.LastUpdatedAt = DateTime.UtcNow;
                await _context.SaveChangesAsync();
            }
        }

        public async Task<bool> CanResetHwidAsync(Guid accountId)
        {
            var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            if (account == null) return false;
            
            if (!account.LastUpdatedAt.HasValue) return true;
            
            
            return (DateTime.UtcNow - account.LastUpdatedAt.Value).TotalDays >= 7;
        }
        
        public async Task<IEnumerable<Account>> GetFilteredUserAccountsAsync(Guid userId, string platformFilter, string statusFilter, string accountSettingFilter)
        {
            var accounts = await _context.Accounts
                .Where(a => a.UserId == userId)
                .Include(a => a.AccountSetting)
                .ToListAsync();
                

            if (!string.IsNullOrEmpty(platformFilter))
            {
                accounts = accounts.Where(a => a.Platform == platformFilter).ToList();
            }

            if (!string.IsNullOrEmpty(statusFilter))
            {
                accounts = accounts.Where(a =>
                    a.Status.ToString().Equals(statusFilter, StringComparison.OrdinalIgnoreCase)).ToList();
            }

            if (!string.IsNullOrEmpty(accountSettingFilter) &&
                Guid.TryParse(accountSettingFilter, out var settingId))
            {
                accounts = accounts.Where(a => a.AccountSettingsId == settingId).ToList();
            }
            
            return accounts;
        }
        
        public async Task<(bool IsSuccess, string ErrorMessage)> ApproveAccountAndNotifyAsync(Guid accountId, string title, string vpsName, string affiliatedHwid, string message, string adminEmail)
        {
            try
            {

                if (_notificationService == null)
                    throw new InvalidOperationException("NotificationService is not available. Make sure to use the correct constructor.");
                
                if (_pythonSocketService == null)
                    throw new InvalidOperationException("PythonSocketService is not available. Make sure to use the correct constructor.");
                
                if (_logger == null)
                    throw new InvalidOperationException("Logger is not available. Make sure to use the correct constructor.");


                await AccountReviewAsync(AccountStatus.Active, accountId, vpsName, adminEmail, affiliatedHwid);


                Notification notification = new Notification
                {
                    Id = Guid.NewGuid(),
                    Title = title,
                    Description = message,
                    DateTime = DateTime.Now
                };

                var account = await GetAccountByIdAsync(accountId);
                List<Guid> target = new List<Guid> { account.UserId };

                await _notificationService.CreateNotificationAsync(notification, target);
                

                var startResult = await _pythonSocketService.StartAccountAsync(account.Id.ToString(), account.Affiliated_HWID);
                if (!startResult.IsSuccess)
                {
                    _logger.LogError($"Failed to start account: {startResult.ErrorMessage}");
                    return (true, $"Account approved but failed to start: {startResult.ErrorMessage}");
                }

                return (true, null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in ApproveAccountAndNotifyAsync");
                return (false, ex.Message);
            }
        }
        
        public async Task<IEnumerable<User>> SearchUsersAsync(string searchTerm)
        {
            if (string.IsNullOrWhiteSpace(searchTerm))
                return new List<User>();
                
            return await _context.Users
                .Where(u => u.UserName.Contains(searchTerm) || 
                           u.Email.Contains(searchTerm))
                .Take(20)
                .ToListAsync();
        }
        
        public async Task<IEnumerable<Account>> GetAccountsByUserIdAsync(Guid userId)
        {
            return await _context.Accounts
                .Where(a => a.UserId == userId)
                .Include(a => a.AccountSetting)
                .ToListAsync();
        }
        
        public async Task<(bool IsSuccess, string ErrorMessage)> ChangeAccountHwidAsync(Guid accountId, string newHwid)
        {
            try
            {
                var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
                if (account == null)
                    return (false, "Акаунтът не е намерен");
                    
                account.Affiliated_HWID = newHwid;
                account.LastUpdatedAt = DateTime.UtcNow;
                await _context.SaveChangesAsync();
                
                if (_pythonSocketService != null)
                {
                    var startResult = await _pythonSocketService.StartAccountAsync(account.Id.ToString(), newHwid);
                    if (!startResult.IsSuccess)
                    {
                        _logger?.LogWarning($"HWID updated in database but failed to update in Python API: {startResult.ErrorMessage}");
                        return (true, "HWID е обновен, но промяната може да не е отразена в работещите системи");
                    }
                }
                
                return (true, null);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error in ChangeAccountHwidAsync");
                return (false, $"Грешка при промяна на HWID: {ex.Message}");
            }
        }
    }
}
