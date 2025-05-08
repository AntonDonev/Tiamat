using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Models;

namespace Tiamat.Core.Services.Interfaces
{
    public class ServiceResult
    {
        public bool IsSuccess { get; set; }
        public string Message { get; set; }
        public List<string> Errors { get; set; } = new List<string>();

        public static ServiceResult Success(string message = "Operation completed successfully")
        {
            return new ServiceResult { IsSuccess = true, Message = message };
        }

        public static ServiceResult Failure(string message, List<string> errors = null)
        {
            return new ServiceResult { IsSuccess = false, Message = message, Errors = errors ?? new List<string>() };
        }
    }

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
        Task<(IEnumerable<Notification> notifications, int totalCount, int totalPages)> GetFilteredAndPagedNotificationsAsync(Guid userId, int page = 1, int pageSize = 3, string startDate = null, string endDate = null);
        
        Task<ServiceResult> SendNotificationToTargetsAsync(string title, string description, string targets);
    }
}