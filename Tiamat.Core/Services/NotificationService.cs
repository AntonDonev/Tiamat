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

        public IEnumerable<Notification> GetAllNotifications()
        {
            return _context.Notifications
                .Include(n => n.NotificationUsers)
                    .ThenInclude(nu => nu.User)
                .ToList();
        }

        public Notification GetNotificationById(Guid id)
        {
            return _context.Notifications
                .Include(n => n.NotificationUsers)
                    .ThenInclude(nu => nu.User)
                .FirstOrDefault(n => n.Id == id);
        }

        public void CreateNotification(Notification notification, IEnumerable<Guid> userIds)
        {
            _context.Notifications.Add(notification);

            foreach (var userId in userIds)
            {
                var notificationUser = new NotificationUser
                {
                    NotificationId = notification.Id,
                    UserId = userId
                };
                _context.NotificationUsers.Add(notificationUser);
            }

            _context.SaveChanges();
        }

        public void UpdateNotification(Notification notification, IEnumerable<Guid> userIds)
        {
            var existingNotification = _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefault(n => n.Id == notification.Id);

            if (existingNotification == null)
                return;

            existingNotification.Title = notification.Title;
            existingNotification.Description = notification.Description;
            existingNotification.DateTime = notification.DateTime;

            _context.NotificationUsers.RemoveRange(existingNotification.NotificationUsers);

            // Add new relationships
            foreach (var userId in userIds)
            {
                var notificationUser = new NotificationUser
                {
                    NotificationId = existingNotification.Id,
                    UserId = userId
                };
                _context.NotificationUsers.Add(notificationUser);
            }

            _context.SaveChanges();
        }

        public void DeleteNotification(Guid notificationId)
        {
            var notification = _context.Notifications
                .Include(n => n.NotificationUsers)
                .FirstOrDefault(n => n.Id == notificationId);

            if (notification == null)
                return;

            _context.NotificationUsers.RemoveRange(notification.NotificationUsers);

            _context.Notifications.Remove(notification);

            _context.SaveChanges();
        }

        public IEnumerable<Notification> GetUserNotifications(Guid userId)
        {
            return _context.NotificationUsers
                .Include(nu => nu.Notification)
                .Where(nu => nu.UserId == userId)
                .OrderByDescending(nu => nu.Notification.DateTime)
                .Select(nu => nu.Notification)
                .ToList();
        }

        public IEnumerable<Notification> GetUserUnreadNotifications(Guid userId)
        {
            return _context.NotificationUsers
                .Include(nu => nu.Notification)
                .Where(nu => nu.UserId == userId && !nu.IsRead)
                .OrderByDescending(nu => nu.Notification.DateTime)
                .Select(nu => nu.Notification)
                .ToList();
        }

        public void MarkNotificationAsRead(Guid userId, Guid notificationId)
        {
            var notificationUser = _context.NotificationUsers
                .FirstOrDefault(nu => nu.UserId == userId && nu.NotificationId == notificationId);

            if (notificationUser != null && !notificationUser.IsRead)
            {
                notificationUser.IsRead = true;
                notificationUser.ReadAt = DateTime.UtcNow;
                _context.SaveChanges();
            }
        }

        public void MarkAllNotificationsAsRead(Guid userId)
        {
            var notificationUsers = _context.NotificationUsers
                .Where(nu => nu.UserId == userId && !nu.IsRead)
                .ToList();

            foreach (var nu in notificationUsers)
            {
                nu.IsRead = true;
                nu.ReadAt = DateTime.UtcNow;
            }

            _context.SaveChanges();
        }
    }
}
