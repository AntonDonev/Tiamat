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
        Task<IEnumerable<Notification>> GetAllNotificationsAsync();
        Task<Notification> GetNotificationByIdAsync(Guid id);
        Task CreateNotificationAsync(Notification notification, IEnumerable<Guid> userIds);
        Task UpdateNotificationAsync(Notification notification, IEnumerable<Guid> userIds);
        Task<IEnumerable<NotificationUser>> GetUserNotificationsUserAsync(Guid? userId);
        Task DeleteNotificationAsync(Guid notificationId);
        Task CreateNotificationEveryoneAsync(Notification notification);
        Task<IEnumerable<Notification>> GetUserNotificationsAsync(Guid userId);
        Task<IEnumerable<Notification>> GetUserUnreadNotificationsAsync(Guid userId);
        Task MarkNotificationAsReadAsync(Guid userId, Guid notificationId);
        Task MarkAllNotificationsAsReadAsync(Guid userId);
    }
}