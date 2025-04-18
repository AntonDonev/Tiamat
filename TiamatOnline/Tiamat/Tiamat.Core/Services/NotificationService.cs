using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using Tiamat.Core.Services.Interfaces;
using Tiamat.DataAccess;
using Tiamat.Models;

namespace Tiamat.Core.Services
{
    public class NotificationService : INotificationService
    {
        private readonly TiamatDbContext _context;

        public NotificationService(TiamatDbContext context)
        {
            _context = context;
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
            return await _context.NotificationUsers
                .Include(nu => nu.Notification)
                .Where(nu => nu.UserId == userId)
                .OrderByDescending(nu => nu.Notification.DateTime)
                .Select(nu => nu.Notification)
                .ToListAsync();
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
    }
}