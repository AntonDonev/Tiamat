using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Models;

namespace Tiamat.Core.Services.Interfaces
{
    public interface IPositionService
    {
        Task CreatePositionAsync(string Symbol, string Type, Account account, decimal Size, decimal Risk, DateTime OpenedAt, string Id);
        Task ClosePositionAsync(string Id, decimal profit, decimal currentCapital, DateTime ClosedAt, string FromHwid);
        Task<List<Position>> GetPositionsOfUserAsync(Guid id);
    Task<List<Position>> GetFilteredPositionsForAccountAsync(Guid accountId, DateTime? startDate, DateTime? endDate, string type, string resultFilter);
    }
}
