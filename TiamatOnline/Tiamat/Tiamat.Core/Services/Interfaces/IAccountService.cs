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
        Task AccountReviewAsync(AccountStatus newStatus, Guid accountId, string VPSName, string AdminEmail, string AffiliatedHWID);

        Task AccountReviewAsync(AccountStatus newStatus, Guid accountId);

        Task<IEnumerable<(Guid, string?)>> AllAccountsAsync();
        Task<Account> GetAccountByHwidAsync(string hwid);
        Task<Account> GetAccountWithPositionsAsync(Guid id);
        Task ResetHwidAsync(Guid accountId, string newHwid);
        Task<bool> CanResetHwidAsync(Guid accountId);
        Task<IEnumerable<Account>> GetFilteredUserAccountsAsync(Guid userId, string platformFilter, string statusFilter, string accountSettingFilter);
        Task<(bool IsSuccess, string ErrorMessage)> ApproveAccountAndNotifyAsync(Guid accountId, string title, string vpsName, string affiliatedHwid, string message, string adminEmail);
        Task<(bool IsSuccess, string ErrorMessage)> DenyAccountAndNotifyAsync(Guid accountId, string title, string message, bool useDefaultMessage);
        
        Task<IEnumerable<User>> SearchUsersAsync(string searchTerm);
        Task<IEnumerable<Account>> GetAccountsByUserIdAsync(Guid userId);
        Task<(bool IsSuccess, string ErrorMessage)> ChangeAccountHwidAsync(Guid accountId, string newHwid);
    }
}
