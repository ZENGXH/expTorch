------------------------------------------------------------------------
--[[ PrinterObserver ]]--
-- used to print loss and accuracy in batch_interval
-- need to setup attributes in report: 
-- 'batch_acc' 
-- 'batch_err' 
-- target on the feedback
------------------------------------------------------------------------
local PrinterObserver, parent = torch.class('dp.PrinterObserver' , 'dp.Observer')

function PrinterObserver:__init(config)
    local args = {}
    dp.helper.unpack_config(args, 
    {config},
    'PrinterObserver', 
    'printer to print the infomation needed', 
    {arg='name', type='string', default='printer',
    helper='name of prineter'},
    {arg='display_interval', type='number', req=true, 
    helper='number batch interval to display'}
    )
    parent.__init(self, config)

    self._print_report = {'batch_acc', 'batch_err'}
    self.display_interval = args.display_interval
    self.batch_counter = 0
    self.loss_all = 0
    self.acc_all = 0
    self.loss_interval_table = {}
    self.acc_interval_table = {}
end

function PrinterObserver:doneBatch(report)
    self.batch_counter = self.batch_counter + 1

    local acc = report.batch_acc or 0
    table.insert(self.acc_interval_table, acc)
    self.acc_all = self.acc_all + acc

    local loss = report.batch_error or 0
    if torch.type(loss) == 'table' then
        loss = loss[1]
    end
    table.insert(self.loss_interval_table, loss)
    self.loss_all = self.loss_all + loss

    if self.batch_counter % self.display_interval == 0 then
        self:display()
    end
end

function PrinterObserver:display()
    local ave_acc = torch.Tensor(self.acc_interval_table):mean()
    local ave_loss = torch.Tensor(self.loss_interval_table):mean()
    self.acc_interval_table = {}
    self.loss_interval_table = {}
    self.log:info(
    string.format('acc: %s%.4f%s and loss %.4f', sys.COLORS.red, ave_acc,sys.COLORS.none, ave_loss)
    )
end
