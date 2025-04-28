using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Core.Services.Interfaces;
using Tiamat.DataAccess;
using Tiamat.Models;

namespace Tiamat.Core.Services
{
    public class AccountService : IAccountService
    {
        private readonly TiamatDbContext _context;

        public AccountService(TiamatDbContext context)
        {
            _context = context;
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

        public async Task<Account> GetAccountByIpAsync(string Ip)
        {
            return await _context.Accounts.FirstOrDefaultAsync(x => x.Affiliated_IP == Ip);
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

        public async Task AccountReviewAsync(AccountStatus newStatus, Guid accountId, string VPSName, string AdminEmail, string AffiliatedIP)
        {
            var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            if (account != null)
            {
                account.VPSName = VPSName;
                account.AdminEmail = AdminEmail;
                account.Status = newStatus;
                account.Affiliated_IP = AffiliatedIP;
                await _context.SaveChangesAsync();
            }
        }

        public async Task AccountReviewAsync(AccountStatus newStatus, Guid accountId)
        {
            var account = await _context.Accounts.FirstOrDefaultAsync(a => a.Id == accountId);
            if (account != null)
            {
                account.Status = newStatus;
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
                .Select(x => new { x.Id, x.Affiliated_IP })
                .ToListAsync();

            return accounts.Select(x => (x.Id, x.Affiliated_IP)).ToList();
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
    }
}
