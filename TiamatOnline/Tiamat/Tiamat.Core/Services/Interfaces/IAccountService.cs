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
        Task<IEnumerable<Account>> GetAllAccountsAsync();
        Task<Account> GetAccountByIdAsync(Guid id);
        Task CreateAccountAsync(Account account);
        Task UpdateAccountAsync(Account account);
        Task DeleteAccountAsync(Guid id);
        Task<IEnumerable<Account>> FilterAccountsAsync(string platform, AccountStatus? status, Guid? accountSettingId);
        Task<int> GetActiveAccountsPerUserIdAsync(Guid userId);
        Task AccountReviewAsync(AccountStatus newStatus, Guid accountId, string VPSName, string AdminEmail, string AffiliatedIP);

        Task AccountReviewAsync(AccountStatus newStatus, Guid accountId);

        Task<IEnumerable<(Guid, string?)>> AllAccountsAsync();
        Task<Account> GetAccountByIpAsync(string Ip);
        Task<Account> GetAccountWithPositionsAsync(Guid id);
    }
}
