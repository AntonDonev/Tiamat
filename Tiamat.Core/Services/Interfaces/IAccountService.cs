using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Principal;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Models;

namespace Tiamat.Core.Services.Interfaces
{
    public interface IAccountService
    {
        IEnumerable<Account> GetAllAccounts();
        Account GetAccountById(Guid id);
        void CreateAccount(Account account);
        void UpdateAccount(Account account);
        void DeleteAccount(Guid id);
        IEnumerable<Account> FilterAccounts(string platform, AccountStatus? status, Guid? accountSettingId);
        public int GetActiveAccountsPerUserId(Guid userId);

        public void AccountReview(AccountStatus newStatus, Guid accountId, string VPSName, string AdminEmail);

        public IEnumerable<(Guid, string?)> AllAccounts();
        public Account GetAccountByIp(string Ip);

        public Account GetAccountWithPositions(Guid id);
    }
}
