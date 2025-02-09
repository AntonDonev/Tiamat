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

        public void ChangeAccountStatus(Guid accountId, AccountStatus newStatus);

        public Account GetAccountWithPositions(Guid id);
    }
}
