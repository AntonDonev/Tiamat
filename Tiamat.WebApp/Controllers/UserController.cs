using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using Tiamat.Core.Services;
using Tiamat.Core.Services.Interfaces;
using Tiamat.DataAccess;
using Tiamat.Models;
using Tiamat.Utility;

namespace Tiamat.WebApp.Controllers
{
    public class UserController : Controller
    {
        private readonly SignInManager<User> _signInManager;
        private readonly UserManager<User> _userManager;
        private readonly IAccountService _accountService;
        private readonly IAccountSettingService _accountSettingService;


        public UserController(SignInManager<User> signInManager,
                              UserManager<User> userManager, IAccountService _AS, IAccountSettingService accountSettingService)
        {
            _signInManager = signInManager;
            _userManager = userManager;
            _accountService = _AS;
            _accountSettingService = accountSettingService;
        }

        [HttpGet]
        public IActionResult Login()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Login(string username, string password)
        {
            var result = await _signInManager.PasswordSignInAsync(username, password, false, false);
            if (result.Succeeded)
            {
                return RedirectToAction("Index", "Home");
            }

            ModelState.AddModelError("", "Invalid login attempt.");
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Logout()
        {
            await _signInManager.SignOutAsync();
            return RedirectToAction("Index", "Home");
        }

        [Authorize]
        [HttpGet]
        public IActionResult Dashboard()
        {
            return View();
        }

        [Authorize]
        [HttpGet] 
        public IActionResult AccountCenter(string? PlatformFilter, string? StatusFilter, string? AccountSettingFilter)
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));

            var userSettingsList = _accountSettingService.GetSettingsForUser(userId).ToList();
            ViewBag.AccountSettings = userSettingsList;

            ViewBag.SelectedPlatformFilter = PlatformFilter;
            ViewBag.SelectedStatusFilter = StatusFilter;
            ViewBag.SelectedSettingFilter = AccountSettingFilter;

            var accounts = _accountService.GetAllAccounts();

            if (!string.IsNullOrEmpty(PlatformFilter))
            {
                accounts = accounts.Where(a => a.Platform == PlatformFilter);
            }

            if (!string.IsNullOrEmpty(StatusFilter))
            {
                accounts = accounts.Where(a =>
                    a.Status.ToString().Equals(StatusFilter, StringComparison.OrdinalIgnoreCase));
            }
            if (!string.IsNullOrEmpty(AccountSettingFilter))
            {
                if (Guid.TryParse(AccountSettingFilter, out var settingId))
                {
                    accounts = accounts.Where(a => a.AccountSettingsId == settingId);
                }
            }

            return View(accounts.ToList());
        }



        [Authorize]
        [HttpGet]
        public IActionResult AddAccount()
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));
            var userSettingsList = _accountSettingService.GetSettingsForUser(userId).ToList();
            ViewBag.AccountSettings = userSettingsList;

            return View();
        }

        [Authorize]
        [HttpPost]
        public IActionResult AddAccount(Account account)
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));

            if (!ModelState.IsValid)
            {
                var userSettingsList = _accountSettingService.GetSettingsForUser(userId).ToList();
                ViewBag.AccountSettings = userSettingsList;
                return View(account);
            }

            try
            {
                account.UserId = userId;
                account.Id = Guid.NewGuid();

                account.CreatedAt = DateTime.UtcNow;
                account.Status = AccountStatus.Pending;

                account.HighestCapital = account.InitialCapital;
                account.LowestCapital = account.InitialCapital;

                _accountService.CreateAccount(account);

                return RedirectToAction(nameof(AccountCenter));
            }
            catch (Exception ex)
            {
                ModelState.AddModelError(string.Empty, $"Error creating the account: {ex.Message}");
                var userSettingsList = _accountSettingService.GetSettingsForUser(userId).ToList();
                ViewBag.AccountSettings = userSettingsList;
                return View(account);
            }
        }




    }
}
