using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Models;

namespace Tiamat.Core.Services.Interfaces
{
    public interface INotificationService
    {
        IEnumerable<Notification> GetAllNotifications();
        Notification GetNotificationById(Guid id);
        void CreateNotification(Notification notification, IEnumerable<Guid> userIds);
        void UpdateNotification(Notification notification, IEnumerable<Guid> userIds);
        public IEnumerable<NotificationUser> GetUserNotificationsUser(Guid? userId);
        void DeleteNotification(Guid notificationId);

        IEnumerable<Notification> GetUserNotifications(Guid userId);
        IEnumerable<Notification> GetUserUnreadNotifications(Guid userId);
        void MarkNotificationAsRead(Guid userId, Guid notificationId);
        void MarkAllNotificationsAsRead(Guid userId);
    }
}
