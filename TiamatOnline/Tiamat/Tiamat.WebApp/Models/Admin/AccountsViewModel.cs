using System.ComponentModel.DataAnnotations;
using Tiamat.Models;

namespace Tiamat.WebApp.Models.Admin
{
    public class AccountsViewModel
    {
        public string SearchTerm { get; set; }
        public IEnumerable<User> SearchResults { get; set; }
        public Guid? SelectedUserId { get; set; }
        public string SelectedUserName { get; set; }
        public IEnumerable<Account> UserAccounts { get; set; }
        
        public AccountsViewModel()
        {
            SearchResults = new List<User>();
            UserAccounts = new List<Account>();
        }
    }
    
    public class ChangeHwidViewModel
    {
        public Guid AccountId { get; set; }
        
        [Required(ErrorMessage = "HWID е задължителен")]
        [MaxLength(100, ErrorMessage = "HWID не може да надвишава 100 символа")]
        [Display(Name = "Нов HWID")]
        public string NewHwid { get; set; }
        
        public string AccountName { get; set; }
    }
}