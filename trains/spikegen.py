import torch
from bernoulli import sample

dtype = torch.float

def rate(
    data,
    num_steps=False,
    gain=1,
    offset=0,
    first_spike_time=0,
    time_var_input=False,
):

    """Spike rate encoding of input data. Convert tensor into Poisson spike
    trains using the features as the mean of a
    binomial distribution. If `num_steps` is specified, then the data will be
    first repeated in the first dimension
    before rate encoding.

    If data is time-varying, tensor dimensions use time first.

    Example::

        # 100% chance of spike generation
        a = torch.Tensor([1, 1, 1, 1])
        spikegen.rate(a, num_steps=1)
        >>> tensor([1., 1., 1., 1.])

        # 0% chance of spike generation
        b = torch.Tensor([0, 0, 0, 0])
        spikegen.rate(b, num_steps=1)
        >>> tensor([0., 0., 0., 0.])

        # 50% chance of spike generation per time step
        c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
        spikegen.rate(c, num_steps=1)
        >>> tensor([0., 1., 0., 1.])

        # Increasing num_steps will increase the length of
        # the first dimension (time-first)
        print(c.size())
        >>> torch.Size([1, 4])

        d = spikegen.rate(torch.Tensor([0.5, 0.5, 0.5, 0.5]), num_steps = 2)
        print(d.size())
        >>> torch.Size([2, 4])


    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :param num_steps: Number of time steps. Only specify if input data
        does not already have time dimension, defaults to ``False``
    :type num_steps: int, optional

    :param gain: Scale input features by the gain, defaults to ``1``
    :type gain: float, optional

    :param offset: Shift input features by the offset, defaults to ``0``
    :type offset: torch.optim, optional

    :param first_spike_time: Time to first spike, defaults to ``0``.
    :type first_spike_time: int, optional

    :param time_var_input: Set to ``True`` if input tensor is time-varying.
        Otherwise, `first_spike_time!=0` will modify the wrong dimension.
        Defaults to ``False``
    :type time_var_input: bool, optional

    :return: rate encoding spike train of input features of shape
        [num_steps x batch x input_size]
    :rtype: torch.Tensor

    """

    if first_spike_time < 0 or num_steps < 0:
        raise Exception(
            "``first_spike_time`` and ``num_steps`` cannot be negative."
        )

    if first_spike_time > (num_steps - 1):
        if num_steps:
            raise Exception(
                f"first_spike_time ({first_spike_time}) must be equal to "
                f"or less than num_steps-1 ({num_steps-1})."
            )
        if not time_var_input:
            raise Exception(
                "If the input data is time-varying, set "
                "``time_var_input=True``.\n If the input data is not "
                "time-varying, ensure ``num_steps > 0``."
            )

    if first_spike_time > 0 and not time_var_input and not num_steps:
        raise Exception(
            "``num_steps`` must be specified if both the input is not "
            "time-varying and ``first_spike_time`` is greater than 0."
        )

    if time_var_input and num_steps:
        raise Exception(
            "``num_steps`` should not be specified if input is "
            "time-varying, i.e., ``time_var_input=True``.\n "
            "The first dimension of the input data + ``first_spike_time`` "
            "will determine ``num_steps``."
        )

    device = data.device

    # intended for time-varying input data
    if time_var_input:
        spike_data = rate_conv(data)

        # zeros are added directly to the start of 0th (time) dimension
        if first_spike_time > 0:
            spike_data = torch.cat(
                (
                    torch.zeros(
                        tuple([first_spike_time] + list(spike_data[0].size())),
                        device=device,
                        dtype=dtype,
                    ),
                    spike_data,
                )
            )

    # intended for time-static input data
    else:

        # Generate a tuple: (num_steps, 1..., 1) where the number of 1's
        # = number of dimensions in the original data.
        # Multiply by gain and add offset.
        time_data = (
            data.repeat(
                tuple(
                    [num_steps]
                    + torch.ones(len(data.size()), dtype=int).tolist()
                )
            )
            * gain
            + offset
        )

        spike_data = rate_conv(time_data)

        # zeros are multiplied by the start of the 0th (time) dimension
        if first_spike_time > 0:
            spike_data[0:first_spike_time] = 0

    return spike_data

def rate_conv(data):
    """Convert tensor into Poisson spike trains using the features as
    the mean of a binomial distribution.
    Values outside the range of [0, 1] are clipped so they can be
    treated as probabilities.

        Example::

            # 100% chance of spike generation
            a = torch.Tensor([1, 1, 1, 1])
            spikegen.rate_conv(a)
            >>> tensor([1., 1., 1., 1.])

            # 0% chance of spike generation
            b = torch.Tensor([0, 0, 0, 0])
            spikegen.rate_conv(b)
            >>> tensor([0., 0., 0., 0.])

            # 50% chance of spike generation per time step
            c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
            spikegen.rate_conv(c)
            >>> tensor([0., 1., 0., 1.])

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :return: rate encoding spike train of input features of shape
        [num_steps x batch x input_size]
    :rtype: torch.Tensor

    """

    # Clip all features between 0 and 1 so they can be used as probabilities.
    # TODO: use torch.nn.functional.softmax instead ?!
    clipped_data = torch.clamp(data, min=0, max=1)

    # pass time_data matrix into bernoulli function.
    spike_data = sample.apply(clipped_data)

    return spike_data
