using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.ModelBinding.Metadata;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using Tiamat.Core.Services;
using Tiamat.Core.Services.Interfaces;
using Tiamat.Models;
using Tiamat.Utility;
using Tiamat.Utility.Services;
using Tiamat.WebApp.Models;
using static Tiamat.WebApp.Models.DashboardViewModel;

namespace Tiamat.WebApp.Controllers
{
    [Authorize]
    public class UserController : Controller
    {
        private readonly SignInManager<User> _signInManager;
        private readonly UserManager<User> _userManager;
        private readonly IAccountService _accountService;
        private readonly IAccountSettingService _accountSettingService;
        private readonly INotificationService _notificationService;
        private readonly PythonSocketService _pythonSocketService;
        private readonly IPositionService _positionService;

        public UserController(
            SignInManager<User> signInManager,
            UserManager<User> userManager,
            IAccountService accountService,
            IAccountSettingService accountSettingService,
            INotificationService notificationService,
            PythonSocketService pythonSocketService,
            IPositionService positionService)
        {
            _signInManager = signInManager;
            _userManager = userManager;
            _accountService = accountService;
            _accountSettingService = accountSettingService;
            _notificationService = notificationService;
            _pythonSocketService = pythonSocketService;
            _positionService = positionService;
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Logout()
        {
            await _signInManager.SignOutAsync();
            return RedirectToAction("Index", "Home");
        }

        [HttpGet]
        public async Task<IActionResult> Dashboard(int page = 1, int pageSize = 3)
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));
            var positions = await _positionService.GetPositionsOfUserAsync(userId);

            var sevenDaysAgo = DateTime.Now.AddDays(-7);
            var positionsForChart = positions
                .Select(p => new PositionChartDto
                {
                    Id = p.Id,
                    AccountId = p.AccountId,
                    Type = p.Type,
                    OpenedAtIso = p.OpenedAt.ToString("yyyy-MM-ddTHH:mm:ssZ")
                })
                .ToList();

            System.Diagnostics.Debug.WriteLine($"HELLO! Found {positionsForChart.Count} positions in the last 7 days");
            foreach (var item in positionsForChart)
            {
                System.Diagnostics.Debug.WriteLine($"{item.Id} {item.AccountId} {item.Type} {item.OpenedAtIso}");
            }

            var allNotifications = (await _notificationService.GetAllNotificationsAsync()).ToList();
            int totalPages = (int)Math.Ceiling(allNotifications.Count / (double)pageSize);

            var model = new DashboardViewModel
            {
                Positions = positionsForChart,
                Notifications = allNotifications,
                CurrentPage = page,
                TotalPages = totalPages
            };

            return View(model);
        }

        [HttpGet]
        public async Task<IActionResult> GetNotifications(int page = 1, int pageSize = 3)
        {
            var allNotifications = (await _notificationService.GetAllNotificationsAsync())
                .OrderByDescending(n => n.DateTime)
                .ToList();

            int totalNotifications = allNotifications.Count;
            int totalPages = (int)Math.Ceiling(totalNotifications / (double)pageSize);

            var pagedNotifications = allNotifications
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToList();

            return Json(new
            {
                notifications = pagedNotifications.Select(n => new
                {
                    n.Id,
                    n.Title,
                    n.Description,
                    n.DateTime,
                    n.TotalReadCount
                }),
                currentPage = page,
                totalPages = totalPages
            });
        }

        [HttpGet]
        public async Task<IActionResult> ViewAccount(Guid id)
        {
            var account = await _accountService.GetAccountWithPositionsAsync(id);
            if (account == null) return NotFound();

            var accountSettings = await _accountSettingService.GetSettingsForUserAsync(account.UserId);

            var viewModel = new ViewAccountViewModel
            {
                AccountId = account.Id,
                AccountName = account.AccountName,
                AccountSettingsId = account.AccountSettingsId,
                InitialCapital = account.InitialCapital,
                CurrentCapital = account.CurrentCapital,
                HighestCapital = account.HighestCapital,
                LowestCapital = account.LowestCapital,
                Platform = account.Platform,
                BrokerLogin = account.BrokerLogin,
                BrokerPassword = account.BrokerPassword,
                BrokerServer = account.BrokerServer,
                Status = account.Status,
                VPSName = account.VPSName,
                AdminEmail = account.AdminEmail,
                CreatedAt = account.CreatedAt,
                LastUpdatedAt = account.LastUpdatedAt,
                Positions = account.AccountPositions.Select(ap => new PositionViewModel
                {
                    PositionId = ap.Id,
                    Symbol = ap.Symbol,
                    Size = ap.Size,
                    Type = ap.Type,
                    Risk = ap.Risk,
                    Result = ap.Result,
                    OpenedAt = ap.OpenedAt,
                    ClosedAt = ap.ClosedAt
                }).ToList()
            };

            ViewBag.AccountSettings = accountSettings.ToList();

            return View(viewModel);
        }

        [HttpPost]
        [ServiceFilter(typeof(CheckPythonConnectionAttribute))]
        public async Task<IActionResult> ViewAccount(ViewAccountViewModel model)
        {
            if (!ModelState.IsValid)
            {
                var errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList();

                var combinedErrors = string.Join("; ", errors);

                TempData["AlertMessage"] = "Failed to update account setting: " + combinedErrors;
                TempData["AlertTitle"] = "Validation Error";
                TempData["AlertType"] = "error";
                ViewBag.AccountSettings = (await _accountSettingService.GetSettingsForUserAsync(Guid.Empty)).ToList();

                return RedirectToAction("ViewAccount", new { id = model.AccountId });
            }

            if (!model.AccountSettingsId.HasValue)
            {
                TempData["AlertMessage"] = "Failed to update account setting: ";
                TempData["AlertTitle"] = "Validation Error";
                TempData["AlertType"] = "error";
                return RedirectToAction("ViewAccount", new { id = model.AccountId });
            }

            var account = await _accountService.GetAccountByIdAsync(model.AccountId);
            if (account == null) return NotFound();

            account.AccountName = model.AccountName;
            if (model.AccountSettingsId.HasValue)
            {
                account.AccountSettingsId = model.AccountSettingsId.Value;
                AccountSetting accountSetting = await _accountSettingService.GetSettingByIdAsync(account.AccountSettingsId);
                await _pythonSocketService.EnqueueMessageAsync($"EDIT|{account.Id}|{accountSetting.MaxRiskPerTrade}|{accountSetting.UntradablePeriodMinutes}");
            }

            account.LastUpdatedAt = DateTime.UtcNow;

            await _accountService.UpdateAccountAsync(account);
            TempData["AlertMessage"] = "Account updated successfully!";
            TempData["AlertTitle"] = "Success";
            TempData["AlertType"] = "success";
            return RedirectToAction("ViewAccount", new { id = account.Id });
        }

        [HttpGet]
        public async Task<IActionResult> AccountCenter()
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));

            var settingsForUser = await _accountSettingService.GetSettingsForUserAsync(userId);
            var accountSettingsVm = settingsForUser.Select(s => new AccountSettingViewModel
            {
                AccountSettingId = s.AccountSettingId,
                SettingName = s.SettingName
            }).ToList();

            var accounts = (await _accountService.GetAllAccountsAsync()).Where(a => a.UserId == userId);

            var vm = new AccountCenterViewModel
            {
                PlatformFilter = "",
                StatusFilter = "",
                AccountSettingFilter = "",
                Accounts = accounts.Select(a => new AccountItemViewModel
                {
                    AccountId = a.Id,
                    AccountName = a.AccountName,
                    InitialCapital = a.InitialCapital,
                    HighestCapital = a.HighestCapital,
                    LowestCapital = a.LowestCapital,
                    CurrentCapital = a.CurrentCapital,
                    Platform = a.Platform,
                    Status = a.Status.ToString(),
                    CreatedAt = a.CreatedAt,
                    AccountSettingId = a.AccountSettingsId,
                    AccountSettingName = a.AccountSetting?.SettingName
                }).ToList(),
                AccountSettings = accountSettingsVm
            };

            return View(vm);
        }

        [HttpGet]
        public async Task<IActionResult> Settings()
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));

            var userSettings = await _accountSettingService.GetSettingsForUserAsync(userId);

            var vm = new AccountSettingCenterViewModel
            {
                SettingNameFilter = string.Empty,
                Settings = userSettings.Select(s => new AccountSettingItemViewModel
                {
                    AccountSettingId = s.AccountSettingId,
                    SettingName = s.SettingName,
                    MaxRiskPerTrade = s.MaxRiskPerTrade,
                    UntradablePeriodMinutes = s.UntradablePeriodMinutes
                }).ToList()
            };

            return View(vm);
        }

        [HttpPost]
        public IActionResult Settings(string? settingNameFilter)
        {
            TempData["SettingNameFilter"] = settingNameFilter ?? string.Empty;
            return RedirectToAction(nameof(FilteredSettings));
        }

        [HttpGet]
        public async Task<IActionResult> FilteredSettings()
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));

            var settingNameFilter = TempData["SettingNameFilter"] as string ?? "";

            var userSettings = await _accountSettingService.GetSettingsForUserAsync(userId);

            if (!string.IsNullOrEmpty(settingNameFilter))
            {
                userSettings = userSettings
                    .Where(s => s.SettingName.Contains(settingNameFilter, StringComparison.OrdinalIgnoreCase));
            }

            var vm = new AccountSettingCenterViewModel
            {
                SettingNameFilter = settingNameFilter,
                Settings = userSettings.Select(s => new AccountSettingItemViewModel
                {
                    AccountSettingId = s.AccountSettingId,
                    SettingName = s.SettingName,
                    MaxRiskPerTrade = s.MaxRiskPerTrade,
                    UntradablePeriodMinutes = s.UntradablePeriodMinutes
                }).ToList()
            };

            return View("Settings", vm);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> MarkNotificationAsRead(Guid notificationId)
        {
            var userIdString = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userIdString))
            {
                return Json(new { success = false });
            }

            var userId = Guid.Parse(userIdString);

            await _notificationService.MarkNotificationAsReadAsync(userId, notificationId);

            var newCount = (await _notificationService.GetUserUnreadNotificationsAsync(userId)).Count();

            return Json(new { success = true, unreadCount = newCount });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> MarkAllAsRead()
        {
            var userIdString = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userIdString))
            {
                return Json(new { success = false });
            }

            var userId = Guid.Parse(userIdString);

            await _notificationService.MarkAllNotificationsAsReadAsync(userId);

            var newCount = (await _notificationService.GetUserUnreadNotificationsAsync(userId)).Count();

            return Json(new { success = true, unreadCount = newCount });
        }

        [HttpGet]
        public IActionResult AddAccountSetting()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> AddAccountSetting(AccountSettingAddViewModel vm)
        {
            if (!ModelState.IsValid)
            {
                var errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList();

                var combinedErrors = string.Join("; ", errors);

                TempData["AlertMessage"] = "Failed to create account setting: " + combinedErrors;
                TempData["AlertTitle"] = "Validation Error";
                TempData["AlertType"] = "error";

                return View(vm);
            }
            var setting = new AccountSetting
            {
                AccountSettingId = Guid.NewGuid(),
                SettingName = vm.SettingName,
                MaxRiskPerTrade = vm.MaxRiskPerTrade,
                UntradablePeriodMinutes = vm.UntradablePeriodMinutes,
                UserId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier))
            };
            await _accountSettingService.CreateSettingAsync(setting);
            TempData["AlertMessage"] = "Account created successfully!";
            TempData["AlertTitle"] = "Success";
            TempData["AlertType"] = "success";
            return RedirectToAction(nameof(Settings));
        }

        [HttpPost]
        public IActionResult AccountCenter(string? PlatformFilter, string? StatusFilter, string? AccountSettingFilter)
        {
            TempData["PlatformFilter"] = PlatformFilter ?? string.Empty;
            TempData["StatusFilter"] = StatusFilter ?? string.Empty;
            TempData["AccountSettingFilter"] = AccountSettingFilter ?? string.Empty;

            return RedirectToAction(nameof(FilteredAccountCenter));
        }

        [HttpGet]
        public async Task<IActionResult> FilteredAccountCenter()
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));

            var platformFilter = TempData["PlatformFilter"] as string ?? "";
            var statusFilter = TempData["StatusFilter"] as string ?? "";
            var accountSettingFilter = TempData["AccountSettingFilter"] as string ?? "";

            var settingsForUser = await _accountSettingService.GetSettingsForUserAsync(userId);
            var accountSettingsVm = settingsForUser.Select(s => new AccountSettingViewModel
            {
                AccountSettingId = s.AccountSettingId,
                SettingName = s.SettingName
            }).ToList();

            var accounts = (await _accountService.GetAllAccountsAsync()).Where(a => a.UserId == userId);

            if (!string.IsNullOrEmpty(platformFilter))
            {
                accounts = accounts.Where(a => a.Platform == platformFilter);
            }

            if (!string.IsNullOrEmpty(statusFilter))
            {
                accounts = accounts.Where(a =>
                    a.Status.ToString().Equals(statusFilter, StringComparison.OrdinalIgnoreCase));
            }

            if (!string.IsNullOrEmpty(accountSettingFilter) &&
                Guid.TryParse(accountSettingFilter, out var settingId))
            {
                accounts = accounts.Where(a => a.AccountSettingsId == settingId);
            }

            var accountItemsVm = accounts.Select(a => new AccountItemViewModel
            {
                AccountId = a.Id,
                AccountName = a.AccountName,
                InitialCapital = a.InitialCapital,
                CurrentCapital = a.CurrentCapital,
                HighestCapital = a.HighestCapital,
                LowestCapital = a.LowestCapital,
                Platform = a.Platform,
                Status = a.Status.ToString(),
                CreatedAt = a.CreatedAt,
                AccountSettingId = a.AccountSettingsId,
                AccountSettingName = a.AccountSetting?.SettingName
            }).ToList();

            var vm = new AccountCenterViewModel
            {
                PlatformFilter = platformFilter,
                StatusFilter = statusFilter,
                AccountSettingFilter = accountSettingFilter,
                Accounts = accountItemsVm,
                AccountSettings = accountSettingsVm
            };

            return View("AccountCenter", vm);
        }

        [HttpGet]
        public async Task<IActionResult> AddAccount()
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));
            var userSettingsList = (await _accountSettingService.GetSettingsForUserAsync(userId)).ToList();
            ViewBag.AccountSettings = userSettingsList;

            return View();
        }

        [HttpPost]
        public async Task<IActionResult> AddAccount(Account account)
        {
            var userId = Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));

            if (!ModelState.IsValid)
            {
                var userSettingsList = (await _accountSettingService.GetSettingsForUserAsync(userId)).ToList();
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
                account.CurrentCapital = account.InitialCapital;
                account.Affiliated_IP = null;

                await _accountService.CreateAccountAsync(account);

                return RedirectToAction(nameof(AccountCenter));
            }
            catch (Exception ex)
            {
                ModelState.AddModelError(string.Empty, $"Error creating the account: {ex.Message}");
                var userSettingsList = (await _accountSettingService.GetSettingsForUserAsync(userId)).ToList();
                ViewBag.AccountSettings = userSettingsList;
                return View(account);
            }
        }
    }
}
