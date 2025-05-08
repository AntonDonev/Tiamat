using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Tiamat.Core.Services.Interfaces;
using Tiamat.DataAccess;
using Tiamat.Models;
using Tiamat.Core.Helpers;

namespace Tiamat.Core.Services
{
    public class NotificationService : INotificationService
    {
        private readonly TiamatDbContext _context;
        private readonly UserManager<User> _userManager;

        public NotificationService(TiamatDbContext context, UserManager<User> userManager)
        {
            _context = context;
            _userManager = userManager;
        }

        public async Task<IEnumerable<Notification>> GetAllNotificationsAsync()
        {
            return await _context.Notifications
                .Include(n => n.NotificationUsers)
                    .ThenInclude(nu => nu.User)
                .ToListAsync();
        }

        public async Task<Notification> GetNotificationByIdAsync(Guid id)
        {
            return await _context.Notifications
                .Include(n => n.NotificationUsers)
                    .ThenInclude(nu => nu.User)
                .FirstOrDefaultAsync(n => n.Id == id);
        }

        public async Task CreateNotificationAsync(Notification notification, IEnumerable<Guid> userIds)
        {
            if (notification == null)
                throw new ArgumentNullException(nameof(notification));

            if (userIds == null)
                userIds = Enumerable.Empty<Guid>();

            if (notification.Id == Guid.Empty)
                notification.Id = Guid.NewGuid();

            notification.NotificationUsers = userIds.Select(userId => new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = notification.Id,
                UserId = userId,
                ReadAt = null,
                IsRead = false
            }).ToList();

            notification.DateTime = DateTime.UtcNow;

            await _context.Notifications.AddAsync(notification);
            await _context.SaveChangesAsync();
        }

        public async Task CreateNotificationEveryoneAsync(Notification notification)
        {
            if (notification == null)
                throw new ArgumentNullException(nameof(notification));

            var allUserIds = await _context.Users.Select(u => u.Id).ToListAsync();

            var notificationUsers = allUserIds
                .Select(userId => new NotificationUser
                {
                    Id = Guid.NewGuid(),
                    NotificationId = notification.Id,
                    UserId = userId,
                    ReadAt = null,
                    IsRead = false
                })
                .ToList();

            notification.NotificationUsers = notificationUsers;
            notification.DateTime = DateTime.UtcNow;

            await _context.Notifications.AddAsync(notification);
            await _context.SaveChangesAsync();
        }

        public async Task UpdateNotificationAsync(Notification notification, IEnumerable<Guid> userIds)
        {
            if (notification == null)
                throw new ArgumentNullException(nameof(notification));

            if (userIds == null)
                userIds = Enumerable.Empty<Guid>();

            var existingNotification = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notification.Id);

            if (existingNotification == null)
                return;

            existingNotification.Title = notification.Title;
            existingNotification.Description = notification.Description;
            existingNotification.DateTime = notification.DateTime;

            _context.NotificationUsers.RemoveRange(existingNotification.NotificationUsers);

            existingNotification.NotificationUsers = userIds.Select(userId => new NotificationUser
            {
                Id = Guid.NewGuid(),
                NotificationId = existingNotification.Id,
                UserId = userId
            }).ToList();

            await _context.SaveChangesAsync();
        }

        public async Task DeleteNotificationAsync(Guid notificationId)
        {
            var notification = await _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefaultAsync(n => n.Id == notificationId);

            if (notification == null)
                return;

            _context.NotificationUsers.RemoveRange(notification.NotificationUsers);

            _context.Notifications.Remove(notification);
            await _context.SaveChangesAsync();
        }

        public async Task<IEnumerable<NotificationUser>> GetUserNotificationsUserAsync(Guid? userId)
        {
            return await _context.NotificationUsers
                .Include(nu => nu.Notification)
                .Where(nu => nu.UserId == userId)
                .OrderByDescending(nu => nu.Notification.DateTime)
                .ToListAsync();
        }

        public async Task<IEnumerable<Notification>> GetUserNotificationsAsync(Guid userId)
        {

            
            var userNotifications = await _context.NotificationUsers
                .Include(nu => nu.Notification)
                .Where(nu => nu.UserId == userId)
                .OrderByDescending(nu => nu.Notification.DateTime)
                .Select(nu => nu.Notification)
                .ToListAsync();
                

            return userNotifications;
        }

        public async Task<IEnumerable<Notification>> GetUserUnreadNotificationsAsync(Guid userId)
        {
            return await _context.NotificationUsers
                .Include(nu => nu.Notification)
                .Where(nu => nu.UserId == userId && !nu.IsRead)
                .OrderByDescending(nu => nu.Notification.DateTime)
                .Select(nu => nu.Notification)
                .ToListAsync();
        }

        public async Task MarkNotificationAsReadAsync(Guid userId, Guid notificationId)
        {
            var notificationUser = await _context.NotificationUsers
                .FirstOrDefaultAsync(nu => nu.UserId == userId && nu.NotificationId == notificationId);

            if (notificationUser != null && !notificationUser.IsRead)
            {
                var notification = await _context.Notifications.FirstOrDefaultAsync(x => x.Id == notificationUser.NotificationId);
                if (notification != null)
                {
                    notification.TotalReadCount++;
                }

                notificationUser.IsRead = true;
                notificationUser.ReadAt = DateTime.UtcNow;
                await _context.SaveChangesAsync();
            }
        }

        public async Task MarkAllNotificationsAsReadAsync(Guid userId)
        {
            var notificationUsers = await _context.NotificationUsers
                .Where(nu => nu.UserId == userId && !nu.IsRead)
                .ToListAsync();

            foreach (var nu in notificationUsers)
            {
                nu.IsRead = true;

                var notification = await _context.Notifications.FirstOrDefaultAsync(x => x.Id == nu.NotificationId);
                if (notification != null)
                {
                    notification.TotalReadCount++;
                }

                nu.ReadAt = DateTime.UtcNow;
            }

            await _context.SaveChangesAsync();
        }

        public async Task<(IEnumerable<Notification> notifications, int totalCount, int totalPages)> GetFilteredAndPagedNotificationsAsync(Guid userId, int page = 1, int pageSize = 3, string startDate = null, string endDate = null)
        {
            var userNotifications = await GetUserNotificationsAsync(userId);
            var allNotifications = userNotifications
                .OrderByDescending(n => n.DateTime)
                .ToList();
                
            if (!string.IsNullOrEmpty(startDate) || !string.IsNullOrEmpty(endDate))
            {
                DateTime? parsedStartDate = null;
                DateTime? parsedEndDate = null;
                
                if (!string.IsNullOrEmpty(startDate) && DateTime.TryParse(startDate, out var start))
                {
                    parsedStartDate = start.Date;

                }
                
                if (!string.IsNullOrEmpty(endDate) && DateTime.TryParse(endDate, out var end))
                {
                    parsedEndDate = end.Date.AddDays(1).AddSeconds(-1); 

                }
                
                int beforeFilter = allNotifications.Count;
                
                if (parsedStartDate.HasValue)
                {
                    allNotifications = allNotifications.Where(n => n.DateTime >= parsedStartDate.Value).ToList();

                }
                
                if (parsedEndDate.HasValue)
                {
                    int afterStartFilter = allNotifications.Count;
                    allNotifications = allNotifications.Where(n => n.DateTime <= parsedEndDate.Value).ToList();

                }
            }

            int totalNotifications = allNotifications.Count;
            int totalPages = (int)Math.Ceiling(totalNotifications / (double)pageSize);

            var pagedNotifications = allNotifications
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToList();
                
            foreach (var notification in pagedNotifications)
            {
                if (notification.NotificationUsers == null || !notification.NotificationUsers.Any())
                {
                    notification.NotificationUsers = await _context.NotificationUsers
                        .Where(nu => nu.NotificationId == notification.Id && nu.UserId == userId)
                        .ToListAsync();
                }
            }

            return (pagedNotifications, totalNotifications, totalPages);
        }

        public async Task<ServiceResult> SendNotificationToTargetsAsync(string title, string description, string targets)
        {
            try
            {
                var notification = new Notification
                {
                    Title = title,
                    Description = description
                };

                var mentions = NotificationHelpers.ExtractMentions(targets);
                mentions = mentions.Distinct(StringComparer.OrdinalIgnoreCase).ToList();

                if (mentions.Contains("everyone", StringComparer.OrdinalIgnoreCase))
                {
                    await CreateNotificationEveryoneAsync(notification);
                    return ServiceResult.Success("Нотификацията е изпратена до всички!");
                }

                var userIds = new List<Guid>();

                foreach (var mention in mentions)
                {
                    var byUserName = await _userManager.FindByNameAsync(mention);
                    if (byUserName != null)
                    {
                        userIds.Add(byUserName.Id);
                        continue;
                    }

                    var byEmail = await _userManager.FindByEmailAsync(mention);
                    if (byEmail != null)
                    {
                        userIds.Add(byEmail.Id);
                        continue;
                    }
                }

                userIds = userIds.Distinct().ToList();

                if (userIds.Count > 0)
                {
                    await CreateNotificationAsync(notification, userIds);
                    return ServiceResult.Success($"Нотификацията е изпратена до {userIds.Count} човека!");
                }
                else
                {
                    return ServiceResult.Failure("Няма валидни хора намерени.");
                }
            }
            catch (Exception ex)
            {
                return ServiceResult.Failure("Възникна грешка при изпращане на нотификацията.", 
                    new List<string> { ex.Message });
            }
        }
    }
}