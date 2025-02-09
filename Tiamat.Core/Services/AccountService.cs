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

        public IEnumerable<Account> GetAllAccounts()
        {
            return _context.Accounts
                .Include(a => a.AccountSetting)
                .ToList();
        }

        public Account GetAccountById(Guid id)
        {
            return _context.Accounts
                .Include(a => a.AccountSetting)
                .FirstOrDefault(a => a.Id == id);
        }

        public void CreateAccount(Account account)
        {
            _context.Accounts.Add(account);
            _context.SaveChanges();
        }

        public void UpdateAccount(Account account)
        {
            _context.Accounts.Update(account);
            _context.SaveChanges();
        }

        public void DeleteAccount(Guid id)
        {
            var account = _context.Accounts.FirstOrDefault(a => a.Id == id);
            if (account != null)
            {
                _context.Accounts.Remove(account);
                _context.SaveChanges();
            }
        }

        public IEnumerable<Account> FilterAccounts(string platform, AccountStatus? status, Guid? accountSettingId)
        {
            var query = _context.Accounts.AsQueryable();
            if (!string.IsNullOrEmpty(platform)) query = query.Where(a => a.Platform == platform);
            if (status.HasValue) query = query.Where(a => a.Status == status.Value);
            if (accountSettingId.HasValue) query = query.Where(a => a.AccountSettingsId == accountSettingId.Value);
            return query
                .Include(a => a.AccountSetting)
                .ToList();
        }

        public int GetActiveAccountsPerUserId(Guid userId)
        {
            return _context.Accounts.Where(x => x.UserId == userId && x.Status == AccountStatus.Active).Count();
        }

        public void ChangeAccountStatus(Guid accountId, AccountStatus newStatus)
        {
            var account = _context.Accounts.FirstOrDefault(a => a.Id == accountId);
            if (account != null)
            {
                account.Status = newStatus;
                _context.SaveChanges();
            }
        }

        public Account GetAccountWithPositions(Guid id)
        {
            return _context.Accounts
                .Include(a => a.AccountPositions)
                    .ThenInclude(ap => ap.Position)
                .Include(a => a.AccountSetting)
                .FirstOrDefault(a => a.Id == id);
        }
    }
}
